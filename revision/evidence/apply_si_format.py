# -*- coding: utf-8 -*-
"""
Supplementary formatting.

1. Table S10 was left with every one of its 56 cells filled dark blue (1F4E79)
   and every run white and bold -- an artefact of the landscape rebuild, which
   cloned the header cell's formatting into the body. The fill is removed, text
   returns to automatic colour, bold is kept only on the header row and the
   field-name stub column, and the header row gains a single rule beneath it.
2. The supplementary body font is set to Times New Roman: the document defaults,
   every rFonts in styles.xml, and every run-level rFonts in document.xml,
   counted by where they were changed.
"""
import collections
import re
import sys

sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx                                    # noqa: E402
from lxml import etree                                         # noqa: E402

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
SI = 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'
FONT = 'Times New Roman'

d = Docx(SI)

# ======================================================== 1. Table S10 =========
tbl = None
for t in d.tree.iter(W + 'tbl'):
    if ''.join(x.text or '' for x in t.iter(W + 't')).startswith('FieldSTEP (this work)'):
        tbl = t
assert tbl is not None, 'Table S10 not found'

removed_fill = uncoloured = unbolded = bolded = 0
rows = tbl.findall(W + 'tr')
for ri, tr in enumerate(rows):
    for ci, tc in enumerate(tr.findall(W + 'tc')):
        tcPr = tc.find(W + 'tcPr')
        if tcPr is not None:
            for shd in tcPr.findall(W + 'shd'):
                tcPr.remove(shd)
                removed_fill += 1
            if ri == 0:                                   # a single rule under the header
                tb = tcPr.find(W + 'tcBorders')
                if tb is not None:
                    bot = tb.find(W + 'bottom')
                    if bot is not None:
                        bot.set(W + 'sz', '12')
                        bot.set(W + 'color', '000000')
        header = (ri == 0 or ci == 0)                     # column heads and the stub column
        for r in tc.iter(W + 'r'):
            rPr = r.find(W + 'rPr')
            if rPr is None:
                continue
            col = rPr.find(W + 'color')
            if col is not None and col.get(W + 'val') == 'FFFFFF':
                col.set(W + 'val', 'auto')
                uncoloured += 1
            b = rPr.find(W + 'b')
            if header and b is None:
                rPr.insert(0, etree.SubElement(rPr, W + 'b'))
                bolded += 1
            elif not header and b is not None:
                rPr.remove(b)
                unbolded += 1
                bcs = rPr.find(W + 'bCs')
                if bcs is not None:
                    rPr.remove(bcs)
print(f"Table S10: {removed_fill} fills removed, {uncoloured} white runs set to automatic, "
      f"{unbolded} body runs un-bolded, {bolded} header runs bolded, "
      f"{len(rows[0].findall(W + 'tc'))} header cells ruled")

# ============================================================ 2. the font ======
counts = collections.Counter()


def set_font(el):
    before = (el.get(W + 'ascii'), el.get(W + 'hAnsi'), el.get(W + 'eastAsia'), el.get(W + 'cs'))
    for attr in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        el.set(W + attr, FONT)
    return before


doc_changed = collections.Counter()
for rf in d.tree.iter(W + 'rFonts'):
    doc_changed[set_font(rf)] += 1
counts['document.xml'] = sum(doc_changed.values())

styles = etree.fromstring(d.blobs['word/styles.xml'])
sty_changed = collections.Counter()
for rf in styles.iter(W + 'rFonts'):
    sty_changed[set_font(rf)] += 1
counts['styles.xml'] = sum(sty_changed.values())
d.blobs['word/styles.xml'] = etree.tostring(
    styles, xml_declaration=True, encoding='UTF-8', standalone=True)

print("\nrFonts rewritten:")
for part, n in counts.items():
    print(f"  {part}: {n}")
print("  document.xml, by the value replaced:")
for k, v in doc_changed.most_common():
    print(f"    {k} -> {FONT}  x{v}")
print("  styles.xml, by the value replaced:")
for k, v in sty_changed.most_common():
    print(f"    {k} -> {FONT}  x{v}")

d.save()

# ------------------------------------------------------------------ verify ----
chk = Docx(SI)
names = set()
for part in ('word/document.xml', 'word/styles.xml', 'word/footnotes.xml'):
    blob = chk.blobs.get(part, b'').decode('utf8', 'replace')
    # only inside <w:rFonts .../> -- w:lang also carries an eastAsia attribute
    for tag in re.findall(r'<w:rFonts\b[^>]*/?>', blob):
        names |= set(re.findall(r'w:(?:ascii|hAnsi|eastAsia|cs)="([^"]+)"', tag))
print("\nfont names referenced anywhere in document/styles/footnotes:", sorted(names))
assert names == {FONT}, names
theme = chk.blobs['word/theme/theme1.xml'].decode('utf8', 'replace')
assert 'asciiTheme' not in chk.blobs['word/styles.xml'].decode('utf8', 'replace'), \
    'a style still resolves its font through the theme'
t2 = None
for t in chk.tree.iter(W + 'tbl'):
    if ''.join(x.text or '' for x in t.iter(W + 't')).startswith('FieldSTEP (this work)'):
        t2 = t
# run-level <w:shd w:fill="auto"> is "no fill" and is left alone; what had to go
# is the cell-level 1F4E79.
fills = {s.get(W + 'fill') for s in t2.iter(W + 'shd')}
assert fills <= {'auto', None}, fills
assert not [c for c in t2.iter(W + 'color') if c.get(W + 'val') == 'FFFFFF'], 'white text left'
print("Table S10: no shading, no white text")
