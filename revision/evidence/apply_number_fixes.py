# -*- coding: utf-8 -*-
"""
N1-N6 and the two derived numbers, from NUMBERS_PASS.md section 11.2.

Note on counts: section 11.2 says N1 is in six places; the scan finds it in
**five** (paragraphs 10, 189, 191, 193, 255), and N4 in **four**, not three --
Table 2's STEP (Full) PR-AUC cell was missed on the first pass.  Both counts are
asserted below rather than trusted.

The apostrophe in this document is sometimes U+2019 and sometimes U+0027, so no
match string here spans one.
"""
import sys
sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx                                    # noqa: E402

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
d = Docx(MS)

# -- N1: melanoma full set, seed 42 (0.910) -> three-seed mean 0.913 +/- 0.011 --
d.replace_in_paragraph(10, "pooled 0.910 reflects overlap",
                       "pooled 0.913 ± 0.011 reflects overlap", label='N1 abstract')
d.replace_in_paragraph(189, "melanoma ROC-AUC (0.910; silhouette",
                       "melanoma ROC-AUC (0.913; silhouette", label='N1 Fig 3 caption')
d.replace_in_paragraph(191, "The drop from 0.910 to 0.591 on melanoma",
                       "The drop from 0.913 ± 0.011 to 0.591 on melanoma",
                       label='N1 Results, overlap paragraph')
d.replace_in_paragraph(193, "high ROC-AUC of 0.910.",
                       "high ROC-AUC of 0.913 ± 0.011.", label='N1 Results, UMAP')
d.replace_in_paragraph(255, "reached ROC-AUC 0.910 on 1,582",
                       "reached ROC-AUC 0.913 ± 0.011 on 1,582", label='N1 Discussion')

# -- N2: lung full set, seed 42 (0.652) -> 0.670 +/- 0.016 -------------------
d.replace_in_paragraph(193, "lower ROC-AUC (0.652)", "lower ROC-AUC (0.670 ± 0.016)",
                       label='N2 Results, UMAP')

# -- N3: non-clinical per-epitope mean, v1 basis -> v2 ensemble --------------
d.replace_in_paragraph(187, "non-clinical mean of 0.769 (n = 335;",
                       "non-clinical mean of 0.754 (n = 337;", label='N3 Results, clinical')

# -- N4: STEP PR-AUC rounds down (0.74646) ----------------------------------
d.replace("0.747 ± 0.011", "0.746 ± 0.011", expected=4,
          label='N4 PR-AUC, 4 places (text, Table 1, ablation, Table 2)')

# -- N6: melanoma per-epitope mean, seed 42 -> three-seed mean --------------
# 0.6716 / 0.6755 / 0.7754 -> 0.7075.  The SD (0.059) is left out of the
# abstract to hold the 245-word count; the pooled-minus-per-epitope gap is
# positive for every seed (0.239 / 0.249 / 0.129), so the claim holds regardless.
d.replace_in_paragraph(10, "per-epitope mean of 0.672.", "per-epitope mean of 0.708.",
                       label='N6 abstract')
# Pays for the two words N1 added, with no claim lost.
d.replace_in_paragraph(10, "initiating the adaptive immune response.",
                       "initiating adaptive immunity.", label='abstract, hold 245 words')

# -- the 300-epitope drop, recomputed against the corrected full model ------
d.replace_in_paragraph(202, "to 0.686 (−9.1%), confirming",
                       "to 0.686 (−8.2% relative to the corrected full model, though the "
                       "two were trained under different pipelines), confirming",
                       label='300-epitope drop, -9.1% -> -8.2%')

# -- N5: say which seed the clustered comparison was computed on ------------
d.replace_in_paragraph(157, "The 4.2-percentage-point margin over DAISY is notable:",
                       "The margin over DAISY is notable:", label='N5 Results, drop bare 4.2')
d.replace_in_paragraph(
    157,
    "no experimental structural prior. The margin survives epitope-level resampling, but",
    "no experimental structural prior. The epitope-level comparison was computed on a single "
    "training seed, for which the margin is 4.2 percentage points; between the three-seed means "
    "it is 4.6 points. The margin survives epitope-level resampling, but",
    label='N5 Results, both margins')
d.replace_in_paragraph(
    254,
    "margin over DAISY is 4.2 percentage points (epitope-level cluster bootstrap",
    "margin over DAISY is 4.2 percentage points for the single training seed on which the "
    "epitope-level comparison was computed (cluster bootstrap",
    label='N5 Discussion, seed named')
d.replace_in_paragraph(
    254,
    "across 576 test epitopes), so the margin itself is robust",
    "across 576 test epitopes); between the three-seed means it is 4.6 points, so the margin "
    "itself is robust",
    label='N5 Discussion, mean margin')

d.save()

for label, old, new, n in d.log:
    print(f"  OK  {label}: {n}")
chk = Docx(MS)
print("\nabstract words:", len(chk.ptext(list(chk.paragraphs())[10]).split()))
