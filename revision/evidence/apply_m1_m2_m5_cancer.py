"""
Fix the urgent mismatches from NUMBERS_PASS.md: M1, M2, M5, and carry the
Abstract's cancer numbers into the Results and Discussion.

No new numbers are introduced beyond those already established in
NUMBERS_PASS.md, R2-5_cancer_memorisation.md and R1-5_delong.md.

Read-only on revision/as_submitted/.
"""
import copy
import sys

sys.path.insert(0, 'revision/evidence')
from lxml import etree

from docx_tools import Docx, W, WT

WR = f'{{{W}}}r'
WRPR = f'{{{W}}}rPr'
WVA = f'{{{W}}}vertAlign'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
EM, RSQ, TIMES = '—', '’', '×'
SUP3, SUP4, SUPMIN = '³', '⁴', '⁻'

d = Docx(MS)
before = [d.ptext(p) for p in d.paragraphs()]


def para(needle):
    hits = [i for i, t in enumerate(before) if needle in t]
    assert len(hits) == 1, f'{len(hits)} paragraphs contain {needle[:50]!r}'
    return hits[0]


# ===================================================== M1 + M5
# The paragraph carried the superseded baseline list alongside the corrected
# values four sentences later. The list is replaced, ATM-TCR and ERGO-LSTM are
# reported as a tie (delta 0.0020, CI [-0.0363, +0.0406], P = 0.91), and the
# significance claim moves to the epitope-level cluster bootstrap, under which
# STEP - DAISY is 2.80e-3 and NOT below 0.001.
d.replace(
    'STEP significantly outperformed all five baselines, DAISY (ROC-AUC 0.701), '
    'ERGO-LSTM (0.589), ERGO-AE (0.570), TEINet (0.493) and ATM-TCR (0.479) with '
    f'every pairwise comparison significant by DeLong\'s test (P < 0.001; STEP '
    f'versus DAISY, P = 1.32 {TIMES} 10⁻⁵; Supplementary Table S3).',
    'STEP outperformed all five baselines: DAISY (ROC-AUC 0.701), ERGO-AE (0.634), '
    'ATM-TCR (0.585), ERGO-LSTM (0.583) and TEINet (0.524). ATM-TCR and ERGO-LSTM '
    'differ by 0.002 (95% CI −0.036 to +0.041) and are not separable, so no rank '
    'is placed between them. Under an epitope-level cluster bootstrap over the 576 '
    f'test epitopes every comparison with STEP is significant, at P < 10{SUPMIN}{SUP4} '
    f'against the four weaker baselines and P = 2.80 {TIMES} 10{SUPMIN}{SUP3} against '
    'DAISY (Supplementary Table S3).',
    expected=1, label='M1+M5 Results, baseline list and significance')

# ===================================================== M2 + M3
d.replace(
    'The 5.4-percentage-point margin over DAISY is notable: DAISY is the strongest '
    'baseline and, like STEP, a physicochemically-driven model—but it '
    'incorporates no experimental structural prior. The gap therefore points to '
    # NB: this sentence uses a STRAIGHT apostrophe, unlike "component's" in the
    # ablation paragraph. The document mixes U+0027 and U+2019.
    "STEP's contact-prior formulation as the source of much of the gain.",
    'The 4.2-percentage-point margin over DAISY is notable: DAISY is the strongest '
    'baseline and, like STEP, a physicochemically-driven model—but it '
    'incorporates no experimental structural prior. The margin survives '
    'epitope-level resampling, but its source is not isolated by the present '
    'ablation, in which five prior configurations fall within seed variance '
    '(Fig. 4c,d).',
    expected=1, label='M2+M3 Results, margin attribution')

# ===================================================== cancer, Results
d.replace(
    'On the melanoma overlap-free subset, STEP retained statistically above-chance '
    'discrimination with a ROC-AUC of 0.680 (95% CI 0.584–0.765; n = 132). On '
    'the lung overlap-free subset, STEP reached a ROC-AUC of 0.615 (95% CI '
    '0.498–0.728; n = 96); because the lower confidence bound crosses 0.5, we '
    'cannot conclude statistical superiority over random in this small subset and '
    'report the result transparently.',
    'On the melanoma overlap-free subset, STEP reached a ROC-AUC of 0.610 '
    '(n = 136). On the lung overlap-free subset it reached 0.465 (n = 96), below '
    f'chance. The lung result is uninformative rather than weak: at receptor level '
    f'that evaluation cannot be computed at all, because every lung negative uses a '
    f'CDR3β seen in training, so removing training receptors leaves no negatives '
    f'and no ROC-AUC exists.',
    expected=1, label='cancer Results, overlap-free values')

d.replace(
    'The drop from 0.913 to 0.680 on melanoma after removing overlapping pairs',
    'The drop from 0.910 to 0.610 on melanoma after removing overlapping pairs',
    expected=1, label='cancer Results, the drop')
d.replace('the overlap-free estimate of 0.680 therefore represents',
          'the overlap-free estimate of 0.610 therefore represents',
          expected=1, label='cancer Results, conservative reference')

# ===================================================== cancer, UMAP paragraph
i = para('internal representations for cancer antigens')
d.replace_in_paragraph(i, 'consistent with the high ROC-AUC of 0.913',
                       'consistent with the high ROC-AUC of 0.910',
                       label='cancer Results, UMAP melanoma')
d.replace_in_paragraph(i, 'reflecting both the lower ROC-AUC (0.719)',
                       'reflecting both the lower ROC-AUC (0.652)',
                       label='cancer Results, UMAP lung')

# ===================================================== cancer, Fig 3 caption
i = para('clusters is consistent with the high melanoma ROC-AUC (0.913')
d.replace_in_paragraph(i, 'melanoma ROC-AUC (0.913', 'melanoma ROC-AUC (0.910',
                       label='Fig 3 caption, melanoma')

# ===================================================== cancer, Discussion
d.replace(
    'STEP retained statistically above-chance discrimination for melanoma (ROC-AUC '
    '0.680, 95% CI 0.584–0.765), but the lung overlap-free confidence interval '
    '(0.498-0.728) crosses chance level, indicating that claims for lung require '
    'larger validation cohorts.',
    'melanoma falls to ROC-AUC 0.610 and lung to 0.465, below chance. The lung '
    'overlap-free evaluation is uninformative rather than weak, because every lung '
    f'negative uses a CDR3β seen in training: removing training receptors leaves '
    'the subset with no negatives at all. Claims for lung require a cohort whose '
    'non-binders are not drawn from the training receptors.',
    expected=1, label='cancer Discussion, limitations')

i = para('reached ROC-AUC 0.913 on 1,582 melanoma')
d.replace_in_paragraph(i, 'reached ROC-AUC 0.913 on 1,582 melanoma',
                       'reached ROC-AUC 0.910 on 1,582 melanoma',
                       label='cancer Discussion, melanoma pooled')

d.save()

# ---------------------------------------------------------------- verify
d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
assert len(after) == len(before)
import difflib
for k, (b, a) in enumerate(zip(before, after)):
    if b != a:
        print(f'\n===== paragraph [{k}] =====')
        sm = difflib.SequenceMatcher(None, b, a, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            if b[i1:i2]:
                print(f'  - {b[i1:i2]}')
            if a[j1:j2]:
                print(f'  + {a[j1:j2]}')

full = '\n'.join(after)
for bad in ('ERGO-LSTM (0.589)', 'ERGO-AE (0.570)', 'TEINet (0.493)',
            'ATM-TCR (0.479)', 'source of much of the gain',
            '5.4-percentage-point', '0.680', '0.615 (95% CI',
            'every pairwise comparison significant by DeLong'):
    assert bad not in full, f'still present: {bad!r}'
print('\nall superseded strings removed')
# Table 1's DAISY Seen-Pair cell is also 0.913 and must NOT have changed
assert sum(t.strip() == '0.913' for t in after) == 1, 'Table 1 cell was touched'
print("Table 1's DAISY seen-pair 0.913 cell untouched")
print('melanoma 0.910 occurrences:', full.count('0.910'))

import zipfile
with zipfile.ZipFile(MS) as z:
    assert z.testzip() is None
print('zip OK, paragraphs', len(after))
