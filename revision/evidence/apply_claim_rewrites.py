"""
Three claim rewrites in revision/working/Manuscript_fv_1_REVISED.docx.

Each replaces a claim the revision's own evidence no longer supports:
  1. the ablation claim      -> R2-7_ablation_provenance.md + R2-4_prior_controls.md
  2. the baseline framing    -> R2-3_baseline_rerun.md + R1-5_delong.md
  3. the cancer claim        -> R2-5_cancer_memorisation.md

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
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
RSQ = '’'          # the document's apostrophe


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
sup_model = next(r for p in ps for r in p.iter(WR)
                 if is_sup(r) and rtext(r) == '53,54')

# ===================================================== 1. THE ABLATION CLAIM
# R2-7: 0.668 came from the trained Full checkpoint scored with
# use_structure_priors=False, i.e. the prior multiplication skipped at
# inference on weights fitted WITH the prior. No prior-free model was ever
# trained. R2-4 retrains five configurations x 3 seeds; all five land within
# seed variance and every across-seed paired CI contains zero.

d.replace(
    'outperforming five state-of-the-art methods (all P < 0.001); ablation '
    'identifies the structural prior as the largest single contributor to '
    'performance.',
    'outperforming five state-of-the-art methods (all P < 0.001); retrained '
    'no-prior, uniform, shuffled and length-only controls, three seeds each, '
    f'place the structural prior{RSQ}s contribution within seed variance.',
    expected=1, label='1a Abstract, ablation clause')

d.replace(
    'Removing structure priors caused the largest drop: ROC-AUC fell from 0.755 '
    'to 0.668 on the balanced set (−11.6%) and from 0.913 to 0.885 on '
    'melanoma. Structure-informed weighting is thus the single most important '
    'architectural component.',
    'The previously reported fall from 0.755 to 0.668 on removing the structure '
    'prior was not a retrained ablation: it was obtained by loading the trained '
    'model and skipping the prior multiplication at inference, so the weights '
    'had been fitted with the prior in place. We therefore retrained the model '
    'from scratch under five prior configurations — the corrected prior, no '
    'prior, a uniform prior, a value-shuffled prior and a length-only mask — '
    'with three seeds each. All five fall within seed variance: ROC-AUC '
    '0.747 ± 0.010 (corrected), 0.754 ± 0.009 (none), 0.756 ± 0.010 '
    '(uniform), 0.751 ± 0.009 (shuffled) and 0.749 ± 0.003 (length-only), '
    'with PR-AUC 0.747 ± 0.011, 0.733 ± 0.010, 0.742 ± 0.016, '
    '0.734 ± 0.023 and 0.740 ± 0.012 respectively. Every across-seed '
    'paired difference between the corrected prior and a control has a 95% '
    'confidence interval containing zero. The prior’s measurable contribution '
    'to accuracy is therefore within run-to-run variation. The melanoma ablation '
    'has not been rerun under these controls [[PENDING: R2-7 — melanoma '
    'ablation, retrained controls]].',
    expected=1, label='1b Results, ablation paragraph')

d.replace(
    'and ablation isolates the structural prior as the largest single contributor '
    'to performance (0.755 → 0.668 on its removal), exceeding the effect of '
    'training-set diversity.',
    'although retrained controls — no prior, uniform, shuffled and length-only, '
    f'three seeds each — place the prior{RSQ}s own contribution to accuracy '
    'within seed variance, so that margin is not attributable to the prior on '
    'present evidence.',
    expected=1, label='1c Discussion, ablation clause')

# ===================================================== 2. THE BASELINE FRAMING
# R2-3: two published values were wrapper artefacts; all six now load
# strict=True with complete key coverage; minimum is 0.5239.
# R1-5 SUPERSEDES "every baseline CI excludes 0.5": under an epitope-level
# cluster bootstrap TEINet's CI is [0.4944, 0.5541] and INCLUDES chance. The
# sample-level claim is not written into the manuscript.

d.replace(
    'Notably, despite this potential advantage, the zero-shot methods perform at '
    'or near chance on unseen epitopes.',
    'In the submitted version two of these baselines scored at or below chance on '
    'unseen epitopes; that was a defect in our evaluation wrappers, not a property '
    'of the models. All six models now load with strict key matching and complete '
    'parameter coverage and produce non-degenerate outputs, and the lowest '
    'baseline ROC-AUC is 0.524.',
    expected=1, label='2a Methods 2.5, zero-shot framing')

# Results sentence: carries a citation, so it is spliced at run level.
OLD = ('Notably, TEINet dropped to near-random performance on unseen epitopes, '
       'underscoring the difficulty of zero-shot epitope generalization for '
       'sequence-only models that lack structural priors.')
NEW_A = ('TEINet is the weakest baseline on unseen epitopes (ROC-AUC 0.524); under '
         'an epitope-level cluster bootstrap its 95% confidence interval, '
         '[0.494, 0.554], includes chance, while the other four baselines remain '
         'above it. The corrected ERGO-AE, ERGO-LSTM and TEINet values (0.634, '
         '0.583 and 0.524) agree to within 0.007 with the epitope-grouped '
         'cross-validation figures reported for those models by Yuan et al.')
NEW_B = ' (0.640, 0.580 and 0.517).'

p165 = next(p for p in d.paragraphs() if OLD in d.ptext(p))
host = next(r for r in p165.iter(WR) if OLD in rtext(r))
ts = [t for t in host.iter(WT)]
assert len(ts) == 1, 'target run is split across <w:t>'
txt = ts[0].text
i = txt.index(OLD)
head, tail = txt[:i], txt[i + len(OLD):]
ts[0].text = head + NEW_A
r_sup = make_run(sup_model, '28')
r_rest = make_run(host, NEW_B + tail)
host.addnext(r_rest)
host.addnext(r_sup)
d.log.append(('2b Results, TEINet sentence', OLD, NEW_A + '28' + NEW_B, 1))

# ===================================================== 3. THE CANCER CLAIM
# R2-5. The submitted "STEP generalized to tumour-associated antigens" was
# already rephrased by the earlier text pass (bb0689e, R2-C3); what remains is
# the bare 0.913 / 0.680 pair, which omits the overlap and is superseded.

d.replace(
    'On tumor-associated antigens, STEP reached ROC-AUC 0.913 (melanoma), 0.680 '
    'with training pairs removed and ranked six clinically important viral '
    'epitopes (CMV, EBV, HIV) with ROC-AUC ≥ 0.963 on the imbalanced test set.',
    'On melanoma a pooled ROC-AUC of 0.910 reflects training overlap: 95.1% of '
    'rows carry a CDR3β seen in training and two MART-1 epitopes supply 88.0% '
    'of the positives, against a per-epitope mean of 0.672. Removing overlapping '
    'training pairs leaves melanoma at 0.610 and lung at 0.465, which is '
    'uninformative. STEP ranked six clinically important viral epitopes (CMV, EBV, '
    'HIV) with ROC-AUC ≥ 0.963 on the imbalanced test set.',
    expected=1, label='3 Abstract, cancer claim')

d.save()

# ------------------------------------------------------------------ verify
d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
assert len(after) == len(before), 'paragraph count changed'
print(f'paragraphs unchanged: {len(after)}')

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

for bad in ('single most important architectural component',
            'largest single contributor',
            'at or near chance',
            'dropped to near-random performance',
            'lack structural priors',
            '0.680 with training pairs removed'):
    assert bad not in '\n'.join(after), f'still present: {bad!r}'
print('\nall six target phrases removed')

ab = after[10]
print(f'abstract word count: {len(ab.split())} (was {len(before[10].split())})')

import re
print('markers:', re.findall(r'\[\[[^\]]+\]\]', '\n'.join(after)))
import zipfile
with zipfile.ZipFile(MS) as z:
    assert z.testzip() is None
print('zip OK')
