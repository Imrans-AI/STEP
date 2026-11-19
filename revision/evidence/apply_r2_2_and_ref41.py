"""
Three edits to revision/working/Manuscript_fv_1_REVISED.docx:

  1. ref 41 in 3.2 is literal body text ("...neural network41..."), the only
     citation in the manuscript that is not a vertAlign="superscript" run. Split
     the run and superscript the number.
  2. R2-2: resolve the [[PENDING: R2-2]] marker in 2.1 Data Splitting by stating
     what the validation set is used for.
  3. R2-2: new "Threshold Selection." paragraph in 2.6, after the Youden's J
     equation, stating that the threshold is fitted on validation and not on test.

Read-only on revision/as_submitted/. Writes only the working copy.
"""
import copy
import sys

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
    r = etree.Element(WR, nsmap=model.nsmap)
    pr = model.find(WRPR)
    if pr is not None:
        r.append(copy.deepcopy(pr))
    t = etree.SubElement(r, WT)
    t.text = text
    if text != text.strip():
        t.set(XML_SPACE, 'preserve')
    return r


d = Docx(MS)
ps = list(d.paragraphs())
before = [d.ptext(p) for p in ps]
sup_model = [r for r in ps[305].iter(WR) if is_sup(r)][0]

# ========================================================= 1. ref 41 -> superscript
p54 = ps[54]
target = [r for r in runs(p54) if 'neural network41' in rtext(r)]
assert len(target) == 1, f'{len(target)} runs hold "neural network41"'
target = target[0]
ts = [t for t in target.iter(WT)]
assert len(ts) == 1
txt = ts[0].text
i = txt.index('network41') + len('network')
head, num, tail = txt[:i], '41', txt[i + 2:]
assert txt[i:i + 2] == '41' and not tail[:1].isdigit(), 'citation boundary'
ts[0].text = head
r_sup = make_run(sup_model, num)
r_tail = make_run(target, tail)
target.addnext(r_tail)
target.addnext(r_sup)
# 3.2 already carries one EMPTY superscript run (a leftover with no text) plus
# the ref 42 run, so the superscript texts must now read ['41', '', '42'].
got = [rtext(r) for r in runs(p54) if is_sup(r)]
assert got == ['41', '', '42'], f'unexpected superscript runs in 3.2: {got}'

# ================================================= 2. R2-2 marker in 2.1
OLD = ('epitope-disjoint from both the training and the test sets, whose role in '
       'model selection is described in Section 2.4 [[PENDING: R2-2]];')
NEW = ('epitope-disjoint from both the training and the test sets and sharing no '
       'TCR–epitope pair with either, used for checkpoint selection and '
       'learning-rate scheduling (Section 2.4) and for fitting the classification '
       'threshold (Section 2.6), and never for reporting performance;')
d.replace(OLD, NEW, expected=1, label='R2-2 marker, 2.1 Data Splitting')

# ================================================= 3. Threshold Selection, 2.6
ps = list(d.paragraphs())
eqn = ps[122]
assert d.ptext(eqn) == '' and len(list(
    eqn.iter('{http://schemas.openxmlformats.org/officeDocument/2006/math}t'))) == 10, \
    "2.6 Youden's J equation paragraph moved"
ci = ps[123]
assert d.ptext(ci).startswith('Confidence Intervals.'), '2.6 layout moved'

newp = etree.Element(WP, nsmap=ci.nsmap)
newp.append(copy.deepcopy(ci.find(WPPR)))
bold_model, body_model = runs(ci)[0], runs(ci)[1]
THRESH = (' The threshold was fitted on the validation set and not on the test '
          'set: it was chosen to maximise Youden’s J on validation '
          'predictions and then applied unchanged to the test set, so that no '
          'test label enters its selection. Because the validation set is small '
          'and strongly imbalanced, the fitted threshold is unstable across '
          'training seeds (0.091, 0.100 and 0.439 for seeds 42, 43 and 44). The '
          'threshold-independent ROC-AUC and PR-AUC therefore remain the primary '
          'metrics, and threshold-dependent values are reported as the mean over '
          'the three training seeds.')
newp.append(make_run(bold_model, 'Threshold Selection.'))
newp.append(make_run(body_model, THRESH))
ci.addprevious(newp)

d.save()

# ----------------------------------------------------------------- verify
d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
print(f'paragraphs {len(before)} -> {len(after)} (expected +1)')
assert len(after) == len(before) + 1

print('\n=== changed / new paragraphs ===')
for i, t in enumerate(after):
    j = i if i < 123 else i - 1
    if i == 123:
        print(f'\n[NEW {i}] + {t}')
    elif before[j] != t:
        print(f'\n[{j} -> {i}]\n  - {before[j][:600]}\n  + {t[:600]}')

print('\n=== superscript runs in 3.2 and the new text ===')
ps2 = list(d2.paragraphs())
for r in ps2[54].iter(WR):
    t = rtext(r)
    if t:
        print(('  SUP ' if is_sup(r) else '      ') + repr(t[-70:]))

assert 'PENDING: R2-2' not in '\n'.join(after), 'marker still present'
print('\n[[PENDING: R2-2]] resolved;',
      sum('PENDING' in t for t in after), 'PENDING paragraph(s) left')

import zipfile
with zipfile.ZipFile(MS) as z:
    assert z.testzip() is None
print('zip OK')
