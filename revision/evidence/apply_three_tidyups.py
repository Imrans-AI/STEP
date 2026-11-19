# -*- coding: utf-8 -*-
"""Three follow-ups to the N-series pass.

1. The melanoma per-epitope mean in the abstract gets its SD (0.708 -> 0.708 +/- 0.059);
   the abstract goes to 247 words, inside the journal's 250-word limit.
2. Paragraph 203's "0.763 versus 0.755" is restated rather than recomputed: both numbers
   come from the original pipeline, so the comparison between them is sound and it is the
   cross-pipeline reading that is not.  Recomputing 0.763 against the corrected 0.747 would
   have compared two models trained differently.
3. Paragraph 254 prints the same P twice at different precision; both now read 2.8 x 10^-3.
"""
import sys
sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx                                    # noqa: E402

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
d = Docx(MS)

# 1 -------------------------------------------------------------------------
d.replace_in_paragraph(10, "per-epitope mean of 0.708.",
                       "per-epitope mean of 0.708 ± 0.059.", label='abstract SD on 0.708')

# 2 -------------------------------------------------------------------------
d.replace_in_paragraph(
    203,
    "marginally raised ROC-AUC (0.763 versus 0.755), but the difference lies within the "
    "95% confidence interval of the full model and is not significant.",
    "marginally raised ROC-AUC, to 0.763 against 0.755 for the full model trained under the "
    "same original pipeline (Table 2); the two are comparable with each other but not with "
    "the corrected full model, which reaches 0.747 ± 0.010. The difference lies within "
    "the 95% confidence interval of the full model and is not significant.",
    label='¶203 bio-negative comparison restated')

# 3 -------------------------------------------------------------------------
d.replace_in_paragraph(254, "P = 2.80 × 10⁻³", "P = 2.8 × 10⁻³",
                       label='¶254 P precision')

d.save()
for label, old, new, n in d.log:
    print(f"  OK  {label}: {n}")

chk = Docx(MS)
cps = [chk.ptext(p) for p in chk.paragraphs()]
print("\nabstract words:", len(cps[10].split()))
assert cps[254].count("P = 2.8 × 10⁻³") == 2, "para 254 P values not uniform"
assert "0.763 versus 0.755" not in cps[203], "old phrasing survives"
assert "same original pipeline" in cps[203]
print("¶203:", cps[203])
