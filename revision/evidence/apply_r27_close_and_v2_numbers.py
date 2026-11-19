"""
Three items:
  1. close [[PENDING: R2-7]] with the two-sided melanoma ablation result
  2. replace the balanced overlap-free figures (0.610 / 0.465) with the
     unbalanced ones (0.591 +/- 0.019, 0.477 +/- 0.027) in the Abstract,
     Results and Discussion together, and record the reason in Methods 2.8
  3. move the per-epitope figures to v2 (342 epitopes, ensemble 0.758)

Sources: NUMBERS_PASS.md §8. Read-only on revision/as_submitted/.
"""
import sys

sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
PM = '±'
d = Docx(MS)
before = [d.ptext(p) for p in d.paragraphs()]


def para(needle):
    hits = [i for i, t in enumerate(before) if needle in t]
    assert len(hits) == 1, f'{len(hits)} paragraphs contain {needle[:60]!r}'
    return hits[0]


# ================================================= 1. close [[PENDING: R2-7]]
# The ablation sentence reports the UNBALANCED overlap-free figure (0.591) and
# not the balanced draw (0.616), so that it agrees with the overlap paragraph
# and the Abstract, which item 2 moves onto the unbalanced subset. Both give the
# same ordering: the corrected prior is last of five either way.
d.replace(
    'The melanoma ablation has not been rerun under these controls '
    '[[PENDING: R2-7 — melanoma ablation, retrained controls]].',
    f'On the melanoma set the same five configurations separate further: the '
    f'corrected prior leads at ROC-AUC 0.913 {PM} 0.011 against 0.894 {PM} 0.006 '
    f'with no prior, and the spread of configuration means is 2.05 times the '
    f'pooled within-configuration standard deviation, against 0.99 on the '
    f'balanced test set. That ordering tracks the training overlap rather than '
    f'generalization — 95.1% of melanoma rows carry a CDR3β seen in '
    f'training — and it reverses once overlapping pairs are removed, where '
    f'the corrected prior is last of the five (0.591 {PM} 0.019, against '
    f'0.635 {PM} 0.012 with no prior). On lung the shuffled prior ranks first.',
    expected=1, label='1 Results, close R2-7')

# ================================================= 2. overlap-free figures
d.replace(
    'Removing overlapping training pairs leaves melanoma at 0.610 and lung at '
    '0.465, which is uninformative.',
    f'Removing overlapping training pairs leaves melanoma at 0.591 {PM} 0.019 and '
    f'lung at 0.477 {PM} 0.027, which is uninformative.',
    expected=1, label='2a Abstract, overlap-free')

d.replace(
    'On the melanoma overlap-free subset, STEP reached a ROC-AUC of 0.610 '
    '(n = 136). On the lung overlap-free subset it reached 0.465 (n = 96), below '
    'chance.',
    f'On the melanoma overlap-free subset, STEP reached a ROC-AUC of '
    f'0.591 {PM} 0.019 over three seeds (n = 847). On the lung overlap-free '
    f'subset it reached 0.477 {PM} 0.027 (n = 130), below chance.',
    expected=1, label='2b Results, overlap-free values')
d.replace('The drop from 0.910 to 0.610 on melanoma',
          'The drop from 0.910 to 0.591 on melanoma',
          expected=1, label='2c Results, the drop')
d.replace('the overlap-free estimate of 0.610 therefore represents',
          'the overlap-free estimate of 0.591 therefore represents',
          expected=1, label='2d Results, conservative reference')
d.replace('performance falls to 0.610 (melanoma) and 0.465 (lung)',
          'performance falls to 0.591 (melanoma) and 0.477 (lung)',
          expected=1, label='2e Results, nearest-neighbour restatement')
d.replace(
    'melanoma falls to ROC-AUC 0.610 and lung to 0.465, below chance.',
    f'melanoma falls to ROC-AUC 0.591 {PM} 0.019 and lung to 0.477 {PM} 0.027, '
    f'below chance.',
    expected=1, label='2f Discussion, limitations')

# Methods 2.8 and 2.1 both carry the sensitivity-analysis sentence. The sizes
# change in both; the explanation of why the subsets are unbalanced belongs only
# in 2.8, which is the section that defines the cancer evaluation.
OLD_SENT = (
    'A sensitivity analysis was performed after removing all TCR\u2013epitope pairs '
    'overlapping with the training set, yielding reduced test sets of 132 melanoma '
    'samples (66+/66\u2212) and 96 lung samples (48+/48\u2212).')
hits = [k for k, t in enumerate(before) if OLD_SENT in t]
assert len(hits) == 2, f'expected 2.1 and 2.8, found {hits}'
i28 = para('The overlap-free analysis therefore measures prediction on unseen')
i21 = [k for k in hits if k != i28][0]

d.replace_in_paragraph(
    i28, OLD_SENT,
    'A sensitivity analysis was performed after removing all TCR\u2013epitope pairs '
    'overlapping with the training set, yielding reduced test sets of 847 melanoma '
    'samples (68+/779\u2212) and 130 lung samples (48+/82\u2212). These subsets are '
    'reported unbalanced. Balancing them by down-sampling the majority class '
    'discards 711 of the 779 melanoma negatives at random, and repeating that draw '
    '2,000 times moves the resulting ROC-AUC over a 95% range of approximately '
    '\u00b10.06 \u2014 wider than the variation between training seeds \u2014 so the '
    'unbalanced subsets, which carry no draw variance, are reported instead.',
    label='2g Methods 2.8, unbalanced subsets')

d.replace_in_paragraph(
    i21, 'yielding reduced test sets of 132 melanoma samples (66+/66\u2212) and 96 '
         'lung samples (48+/48\u2212).',
    'yielding reduced test sets of 847 melanoma samples (68+/779\u2212) and 130 lung '
    'samples (48+/82\u2212); see Section 2.8.',
    label='2g2 Methods 2.1, subset sizes')

# Figure 3f caption carries the old subset sizes
i = para('overlap-free subset (hatched; melanoma n = 132, lung n = 96)')
d.replace_in_paragraph(i, 'overlap-free subset (hatched; melanoma n = 132, lung n = 96)',
                       'overlap-free subset (hatched; melanoma n = 847, lung n = 130)',
                       label='2h Fig 3f caption, subset sizes')

# ================================================= 3. per-epitope, v2
i = para('yielding 340 epitopes from the 581 total')
d.replace_in_paragraph(i, 'yielding 340 epitopes from the 581 total',
                       'yielding 342 epitopes from the 576 total',
                       label='3a Methods 2.7, evaluable count')

i = para('computed for the 340 epitopes with at least five test samples')
d.replace_in_paragraph(
    i, 'computed for the 340 epitopes with at least five test samples, had a mean '
       'of 0.773 ± 0.248',
    'computed for the 342 epitopes with at least five test samples and both classes '
    f'present, had a mean of 0.758 {PM} 0.265',
    label='3b Results, per-epitope mean')
d.replace_in_paragraph(
    i, '42.6% of epitopes reached ROC-AUC ≥ 0.9, 54.1% reached ≥ 0.8 and '
       '66.5% reached ≥ 0.7',
    '42.7% of epitopes reached ROC-AUC ≥ 0.9, 53.2% reached ≥ 0.8 and '
    '64.3% reached ≥ 0.7',
    label='3c Results, per-epitope fractions')

i = para('Distribution of per-epitope ROC-AUC across 340 epitopes')
for old, new, lbl in [
        ('340 epitopes', '342 epitopes', '3d Fig 3 caption, counts (x2)'),
        ('the mean (0.773) and median (0.833)', 'the mean (0.758) and median (0.821)',
         '3e Fig 3a caption, mean and median'),
        ('the dashed line indicates the mean ROC-AUC (0.773)',
         'the dashed line indicates the mean ROC-AUC (0.758)',
         '3f Fig 3b caption, mean')]:
    d.replace_in_paragraph(i, old, new, label=lbl)

d.save()

# ---------------------------------------------------------------- verify
d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
assert len(after) == len(before)
import difflib
for k, (b, a) in enumerate(zip(before, after)):
    if b != a:
        print(f'\n===== paragraph [{k}] =====')
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, b, a, autojunk=False).get_opcodes():
            if tag == 'equal':
                continue
            if b[i1:i2]:
                print(f'  - {b[i1:i2]}')
            if a[j1:j2]:
                print(f'  + {a[j1:j2]}')

full = '\n'.join(after)
for bad in ('PENDING: R2-7', '0.610', '0.465', '340 epitopes', '0.773',
            '42.6%', '54.1%', '66.5%', 'n = 132', 'n = 136'):
    assert bad not in full, f'still present: {bad!r}'
print('\nall superseded strings removed')
import re
print('markers left:', re.findall(r'\[\[[^\]]+\]\]', full))
print('abstract words:', len(after[10].split()))
import zipfile
with zipfile.ZipFile(MS) as z:
    assert z.testzip() is None
print('zip OK, paragraphs', len(after))
