"""
Apply R3-9 (ERGO-II citation, ref 25) and R2-O4 (move ref 55 from Discussion to
Introduction) to revision/working/Manuscript_fv_1_REVISED.docx.

Both items were left unapplied by apply_text_fixes.py because they needed an
author decision on which reference to cite. The user has now supplied the exact
sentences, so only placement and run formatting are decided here.

Citations in this manuscript are vertAlign="superscript" RUNS, not Unicode
superscript characters (67 superscript runs vs 4 stray Unicode superscripts,
all of which belong to "10^-5" style exponents and one earlier edit). The
sentences the user wrote with the glyphs 24/25/55 are therefore built as
superscript runs here, matching every other citation in the document.

Read-only on revision/as_submitted/. Writes only the working copy.
"""
import copy
import re
import sys
from pathlib import Path

sys.path.insert(0, 'revision/evidence')
from lxml import etree

from docx_tools import Docx, W, WT

WR = f'{{{W}}}r'
WRPR = f'{{{W}}}rPr'
WVA = f'{{{W}}}vertAlign'
WP = f'{{{W}}}p'
WPPR = f'{{{W}}}pPr'
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'


# ----------------------------------------------------------------- utilities
def runs(p):
    return [r for r in p.iter(WR)]


def rtext(r):
    return ''.join(t.text or '' for t in r.iter(WT))


def is_sup(r):
    pr = r.find(WRPR)
    if pr is None:
        return False
    va = pr.find(WVA)
    return va is not None and va.get(f'{{{W}}}val') == 'superscript'


def make_run(model, text):
    """A new run carrying `model`'s formatting and `text` as its only <w:t>."""
    r = etree.Element(WR, nsmap=model.nsmap)
    pr = model.find(WRPR)
    if pr is not None:
        r.append(copy.deepcopy(pr))
    t = etree.SubElement(r, WT)
    t.text = text
    if text != text.strip():
        t.set(XML_SPACE, 'preserve')
    return r


def cite_map(doc, lo=10, hi=312):
    """ref number -> [paragraph indices] from superscript citation runs."""
    out = {}
    for i, p in enumerate(doc.paragraphs()):
        if not (lo <= i <= hi):
            continue
        segs = []
        for r in p.iter(WR):
            txt = rtext(r)
            if not txt:
                continue
            s = is_sup(r)
            if segs and segs[-1][0] == s:
                segs[-1][1] += txt
            else:
                segs.append([s, txt])
        for s, txt in segs:
            if not s or not txt.strip():
                continue
            t = txt.strip()
            if not re.fullmatch(r'[0-9,–—\-\s]+', t):
                continue
            for part in t.split(','):
                part = part.strip()
                m = re.fullmatch(r'(\d+)\s*[–—-]\s*(\d+)', part)
                if m:
                    for n in range(int(m.group(1)), int(m.group(2)) + 1):
                        out.setdefault(n, []).append(i)
                elif part.isdigit():
                    out.setdefault(int(part), []).append(i)
    return out


d = Docx(MS)
before_map = cite_map(d)
before_paras = [d.ptext(p) for p in d.paragraphs()]
n_before = len(before_paras)
log = []

# =========================================================== R3-9  (ref 25)
# Placement. The user asked for the sentence "in 2.5 where ERGO is introduced,
# after the existing description". ERGO is described by two items of a
# five-item numbered list (numId 2): DAISY, ERGO-LSTM, ERGO-AE, TEINet,
# ATM-TCR. A prose sentence dropped between ERGO-AE and TEINet would break that
# list in half, and appending it to the ERGO-AE bullet would bury a statement
# about BOTH variants inside the bullet for one of them. It therefore goes in as
# its own body paragraph immediately after the last list item and before the
# "All baselines were obtained..." provenance paragraph - the first prose
# position after the ERGO description.
ps = list(d.paragraphs())
assert ps[112].findtext('.//' + WT, '').startswith('ERGO-AE'), 'list moved'
assert d.ptext(ps[114]).startswith('ATM-TCR'), 'list moved'
anchor = ps[115]
assert d.ptext(anchor).startswith('All baselines were obtained'), 'anchor moved'

newp = etree.Element(WP, nsmap=anchor.nsmap)
newp.append(copy.deepcopy(anchor.find(WPPR)))      # BodyText, 1.5 line, justified
body_model = runs(anchor)[0]
sup_model = [r for r in ps[305].iter(WR) if is_sup(r)][0]

R3_9 = [
    (False, 'We evaluated both variants using the released weights of Springer '
            'et al.'),
    (True, '24'),
    (False, '; the subsequent ERGO-II model'),
    (True, '25'),
    (False, ' extends this framework with MHC and V/J gene context and was not '
            'included, as our benchmark uses CDR3β and epitope sequence alone.'),
]
for sup, txt in R3_9:
    newp.append(make_run(sup_model if sup else body_model, txt))
anchor.addprevious(newp)
log.append(('R3-9', 'new paragraph after the 2.5 baseline list',
            '', ''.join(t for _, t in R3_9)))

# ================================================= R2-O4 (a)  Introduction
# The scarcity of structural data is discussed in the last sentences of the
# Introduction's third paragraph. The new sentence is appended there, after
# "...from the few solved complexes to the many unsolved ones.", where it
# answers the obvious objection (why not predict the structures?) and leads
# into "Here we present STEP". It is NOT placed next to the existing ref 33
# sentence, which already states that structure prediction has advanced.
ps = list(d.paragraphs())
intro = ps[17]
irs = runs(intro)
tail = irs[-1]
tail_t = [t for t in tail.iter(WT)]
assert len(tail_t) == 1 and tail_t[0].text.endswith(
    'to the many unsolved ones.'), 'intro tail run moved'
old_tail = tail_t[0].text

lead = ' Although protein structure prediction has advanced rapidly'
rest = (', predicted models do not resolve the TCR–pMHC interface at the '
        'accuracy required for contact-level inference, and experimentally '
        'determined complexes remain the only reliable source of interface '
        'geometry.')
tail_t[0].text = old_tail + lead
r_sup = make_run(sup_model, '55')
r_rest = make_run(body_model, rest)
tail.addnext(r_rest)
tail.addnext(r_sup)
log.append(('R2-O4a', 'Introduction paragraph [17], appended',
            old_tail[-60:], old_tail[-60:] + lead + '55' + rest))

# ================================================= R2-O4 (b)  Discussion
# ps was refreshed after the insertion, so every index past 115 has moved by
# one; find the Discussion paragraph by its text instead of by position.
hit = [p for p in d.paragraphs()
       if 'single-cell immune-profiling data' in d.ptext(p)]
assert len(hit) == 1, f'{len(hit)} paragraphs match the Discussion target'
disc = hit[0]
drs = runs(disc)
target = [r for r in drs if rtext(r).startswith(', combined with pretraining')][0]
tt = [t for t in target.iter(WT)]
assert len(tt) == 1, 'discussion run split'
OLD = (', combined with pretraining or transfer learning from single-cell '
       'immune-profiling data')
NEW = ', combined with pretraining on large-scale immune repertoire data'
assert tt[0].text == OLD, f'unexpected discussion text: {tt[0].text!r}'
tt[0].text = NEW

sup55 = target.getnext()
assert is_sup(sup55) and rtext(sup55) == '55', 'ref 55 run not where expected'
sup55.getparent().remove(sup55)
log.append(('R2-O4b', 'Discussion paragraph [305]', OLD + '55', NEW))

# ----------------------------------------------------------------- save
d.save()

# ----------------------------------------------------------------- verify
d2 = Docx(MS)
after_paras = [d2.ptext(p) for p in d2.paragraphs()]
after_map = cite_map(d2, 10, 313)       # one paragraph longer now

print('=' * 78)
print('  PARAGRAPH COUNT  %d -> %d  (expected +1)' % (n_before, len(after_paras)))
assert len(after_paras) == n_before + 1

print('\n' + '=' * 78)
print('  CITATION MAP DIFF (superscript citation runs, body only)')
print('=' * 78)
allrefs = sorted(set(before_map) | set(after_map))
for n in allrefs:
    b, a = before_map.get(n, []), after_map.get(n, [])
    if len(b) != len(a) or sorted(b) != sorted([x - 1 if x > 115 else x for x in a]):
        print(f'  ref {n:>2}: before paras {b} -> after paras {a}')
print('  (paragraph indices after 115 shift by +1 because of the new paragraph)')

missing_sup = [n for n in range(1, 58) if n not in after_map]
print(f'\n  refs 1-57 with no superscript citation: {missing_sup}')

full = '\n'.join(after_paras[:314])
for n in missing_sup:
    hits = re.findall(r'[A-Za-z\)\.]' + str(n) + r'(?!\d)', full)
    print(f'    ref {n}: plain-text occurrences in body = {len(hits)} {hits[:5]}')

print('\n' + '=' * 78)
print('  CHANGED PARAGRAPHS')
print('=' * 78)
for i, t in enumerate(after_paras):
    j = i if i < 115 else i - 1
    if i == 115:
        print(f'\n  [NEW {i}] + {t}')
    elif j < n_before and before_paras[j] != t:
        print(f'\n  [{j} -> {i}]')
        print(f'    -  ...{before_paras[j][-420:]}')
        print(f'    +  ...{t[-420:]}')

import zipfile
with zipfile.ZipFile(MS) as z:
    assert z.testzip() is None
print('\n  zip OK, xml parses, %d paragraphs' % len(after_paras))
