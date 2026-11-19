"""
Minimal run-aware .docx text editor used by apply_text_fixes.py.

Word splits a sentence across many <w:r>/<w:t> runs, so a plain string replace on
document.xml misses most targets. These helpers rebuild each paragraph's visible
text, locate the target in that text, and write the replacement back into the runs
it spans, leaving every other part of the package byte-identical.
"""
import re
import shutil
import zipfile
from pathlib import Path

from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
NS = {'w': W, 'm': M}
WT = f'{{{W}}}t'
WP = f'{{{W}}}p'
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'


class Docx:
    def __init__(self, path, part='word/document.xml'):
        self.path = Path(path)
        self.part = part
        with zipfile.ZipFile(self.path) as z:
            self.names = z.namelist()
            self.blobs = {n: z.read(n) for n in self.names}
            self.infos = {i.filename: i for i in z.infolist()}
        self.tree = etree.fromstring(self.blobs[part])
        self.log = []

    # ---------------------------------------------------------------- reading
    def paragraphs(self):
        return self.tree.iter(WP)

    @staticmethod
    def texts(p):
        """All <w:t> in document order, including those inside equations."""
        return [t for t in p.iter(WT)]

    @classmethod
    def ptext(cls, p):
        return ''.join(t.text or '' for t in cls.texts(p))

    def full_text(self):
        return '\n'.join(self.ptext(p) for p in self.paragraphs())

    def find(self, needle):
        """Paragraph indices whose visible text contains needle."""
        return [i for i, p in enumerate(self.paragraphs()) if needle in self.ptext(p)]

    # ---------------------------------------------------------------- writing
    def replace(self, old, new, expected=1, label=''):
        """Replace `old` with `new` wherever it occurs inside a single paragraph."""
        hits = 0
        for p in self.paragraphs():
            ts = self.texts(p)
            if not ts:
                continue
            whole = ''.join(t.text or '' for t in ts)
            if old not in whole:
                continue
            hits += self._splice(ts, old, new)
        if hits != expected:
            raise AssertionError(
                f"{label or old[:40]!r}: expected {expected} replacement(s), made {hits}")
        self.log.append((label, old, new, hits))
        return hits

    @staticmethod
    def _splice(ts, old, new):
        """Write `new` over every occurrence of `old`, touching only the runs the
        match actually spans.

        Rewriting the whole paragraph into its first run would be simpler, but a
        paragraph's <w:t> runs are interleaved with inline OMML equations, whose
        text lives in <m:t> and is not in this list. Collapsing the <w:t> runs
        would therefore move the prose in front of every equation in the
        paragraph. Offsets are kept and only the overlapping runs are edited.
        """
        spans, pos = [], 0
        for t in ts:
            txt = t.text or ''
            spans.append([t, pos, pos + len(txt)])
            pos += len(txt)
        whole = ''.join(t.text or '' for t in ts)

        starts, i = [], whole.find(old)
        while i != -1:
            starts.append(i)
            i = whole.find(old, i + len(old))

        for s in reversed(starts):                 # right to left keeps offsets valid
            e = s + len(old)
            touched = [sp for sp in spans if sp[1] < e and sp[2] > s]
            first = touched[0]
            head = (first[0].text or '')[:s - first[1]]
            last = touched[-1]
            tail = (last[0].text or '')[e - last[1]:] if last[2] > e else ''
            first[0].text = head + new + (tail if last is first else '')
            for sp in touched[1:]:
                sp[0].text = tail if sp is last else ''
            for t, _, _ in spans:
                if t.text and t.text != t.text.strip():
                    t.set(XML_SPACE, 'preserve')
        return len(starts)

    def replace_in_paragraph(self, pidx, old, new, label=''):
        """Same, but confined to one paragraph index (for strings that recur)."""
        p = list(self.paragraphs())[pidx]
        ts = self.texts(p)
        whole = ''.join(t.text or '' for t in ts)
        if old not in whole:
            raise AssertionError(f"{label}: {old[:50]!r} not in paragraph {pidx}")
        n = self._splice(ts, old, new)
        self.log.append((label, old, new, n))
        return n

    def set_math_text(self, pidx, old, new, label=''):
        """Replace the text of an inline equation glyph (<m:t>) in one paragraph."""
        p = list(self.paragraphs())[pidx]
        done = 0
        for t in p.iter(f'{{{M}}}t'):
            if (t.text or '') == old:
                t.text = new
                done += 1
        if done != 1:
            raise AssertionError(f"{label}: matched {done} m:t runs equal to {old!r}")
        self.log.append((label, f"m:t {old!r}", f"m:t {new!r}", 1))

    # ------------------------------------------------------------- equations
    def delimiters(self):
        """Every OMML delimiter object, in document order, with its begin/end
        characters (OMML's defaults are '(' and ')' when the element is absent)."""
        out = []
        for d in self.tree.iter(f'{{{M}}}d'):
            pr = d.find(f'{{{M}}}dPr')
            beg = end = None
            if pr is not None:
                b = pr.find(f'{{{M}}}begChr')
                e = pr.find(f'{{{M}}}endChr')
                beg = b.get(f'{{{M}}}val') if b is not None else None
                end = e.get(f'{{{M}}}val') if e is not None else None
            inner = ''.join(t.text or '' for t in d.iter(f'{{{M}}}t'))
            out.append(dict(el=d, beg='(' if beg is None else beg,
                            end=')' if end is None else end, inner=inner))
        return out

    def set_end_char(self, d, value, label=''):
        """Set (or create) the closing delimiter character of one m:d object."""
        el = d['el']
        pr = el.find(f'{{{M}}}dPr')
        if pr is None:
            pr = etree.SubElement(el, f'{{{M}}}dPr')
            el.insert(0, pr)
        e = pr.find(f'{{{M}}}endChr')
        if e is None:
            e = etree.SubElement(pr, f'{{{M}}}endChr')
            # endChr must precede m:ctrlPr in the schema
            ctrl = pr.find(f'{{{M}}}ctrlPr')
            if ctrl is not None:
                pr.remove(e)
                ctrl.addprevious(e)
        before = d['end']
        e.set(f'{{{M}}}val', value)
        self.log.append((label, f"endChr={before!r}", f"endChr={value!r}", 1))
        return before

    # ---------------------------------------------------------------- saving
    def save(self, out_path=None):
        out = Path(out_path or self.path)
        self.blobs[self.part] = etree.tostring(
            self.tree, xml_declaration=True, encoding='UTF-8', standalone=True)
        tmp = out.with_suffix('.tmp')
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            for n in self.names:
                info = self.infos[n]
                zi = zipfile.ZipInfo(n, date_time=info.date_time)
                zi.compress_type = info.compress_type
                zi.external_attr = info.external_attr
                z.writestr(zi, self.blobs[n])
        shutil.move(str(tmp), str(out))
        return out


def para_dump(path, part='word/document.xml'):
    d = Docx(path, part)
    return [d.ptext(p) for p in d.paragraphs()]


if __name__ == '__main__':
    import sys
    for i, t in enumerate(para_dump(sys.argv[1])):
        if t.strip():
            print(f"[{i}] {t}")
