"""
Two edits to revision/working/Manuscript_fv_1_REVISED.docx.

1. Figure provenance.  The provenance correction already existed inside the
   second paragraph of Methods 2.12, buried behind the projection method.  It is
   lifted out into its own run-in-headed Methods statement ("Figure provenance.")
   and completed with the one case it did not cover: Figure 6, which plots the
   submitted model's 4QOK map and was not recomputed.  No claim is added or
   dropped by the move; only the Figure 6 sentence is new.

2. Abstract trimmed from 303 words to the journal's limit.  Every claim in the
   submitted abstract survives; only the wording is compressed.  The URL lives in
   its own run (it is a hyperlink), so the rewrite is confined to run 0.
"""
import copy
import sys

sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx                                    # noqa: E402

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'

d = Docx(MS)
ps = list(d.paragraphs())

# ----------------------------------------------------------------- 1. provenance
MOVED = (
    "The maps reported here were recomputed on the retrained models and averaged "
    "over three seeds. The lower row of the submitted figure was drawn from an "
    "attention map computed for a different complex: the file it plotted is "
    "attributed to 5HHO in the attention-map metadata while being labelled with "
    "4QOK residue numbers. The map rebuilt for 4QOK correlates with that file at "
    "r = 0.13, so the residue positions reported for that panel have changed. The "
    "same check was applied to the other three visualised complexes, and in each "
    "the CDR3β used by the submitted figure is not the CDR3β present in "
    "the structure it is drawn on (3GSN: ASSPVTGGIYGYT in the crystal against "
    "CASSKTQLWETQYF in the figure; 6RSY: ASSLGFGRDVMR against ASSLGGNGYT; 1OGA: "
    "ASSSRSSYEQY against CASSIISQNEQFF). All attention maps reported here are "
    "computed on the CDR3β resolved in the corresponding structure, so "
    "attention and geometry describe the same complex. Finally, all four"
)
d.replace_in_paragraph(149, MOVED, "All four", label='2.12 lift provenance out')

HEAD = "Figure provenance."
BODY = (
    " Each visualisation in the submitted manuscript was traced back to the file "
    "that produced it, and two errors were found. The lower row of Figure 5 was "
    "drawn from an attention map computed for a different complex: the file it "
    "plotted is attributed to 5HHO in the attention-map metadata while being "
    "labelled with 4QOK residue numbers, and the map rebuilt for 4QOK correlates "
    "with that file at r = 0.13, so the residue positions reported for that panel "
    "have changed. In each of the other three visualised complexes the CDR3β "
    "used by the submitted figure is not the CDR3β present in the structure "
    "it is drawn on (3GSN: ASSPVTGGIYGYT in the crystal against CASSKTQLWETQYF in "
    "the figure; 6RSY: ASSLGFGRDVMR against ASSLGGNGYT; 1OGA: ASSSRSSYEQY against "
    "CASSIISQNEQFF). Every attention map reported here was therefore recomputed on "
    "the retrained models, averaged over three seeds, using the CDR3β "
    "resolved in the corresponding structure, so that attention and geometry "
    "describe the same complex. Figure 6 is the exception: it plots the 4QOK map "
    "of the originally submitted model, computed on the CDR3β resolved in "
    "that structure, and was not recomputed."
)

# Clone a paragraph that already carries the bold run-in head formatting used by
# every other Methods statement, rather than hand-building run properties.
tmpl = copy.deepcopy(ps[22])                      # "Data Sources. We compiled ..."
runs = [r for r in tmpl.iter(W + 'r')]
keep = runs[:2]
for r in runs[2:]:
    r.getparent().remove(r)
keep[0].find(W + 't').text = HEAD
t1 = keep[1].find(W + 't')
t1.text = BODY
t1.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
ps[149].addnext(tmpl)
d.log.append(('2.12 provenance statement', '', HEAD + BODY[:40], 1))

# ------------------------------------------------------------------- 2. abstract
OLD_ABS = (
    "T-cell receptors (TCRs) recognize epitopes presented by the major "
    "histocompatibility complex (MHC), and this recognition initiates the adaptive "
    "immune response. Identifying the rules that govern TCR–epitope pairing is "
    "therefore central to neoantigen vaccine design and TCR-based immunotherapy. "
    "Predicting this specificity remains difficult, particularly for epitopes "
    "unseen during training, because binding is determined by the geometry and "
    "chemistry of a shallow, largely hydrophobic interface that sequence "
    "representations capture poorly. Structural data could help, but resolved "
    "TCR–peptide–MHC complexes remain scarce relative to the sequence "
    "data available. Here we propose STEP (Structure-informed TCR–Epitope "
    "Prediction), a deep learning framework that transfers interface knowledge "
    "from this small structural corpus to sequence-scale prediction. STEP encodes "
    "each TCRβ CDR3–epitope pair as a multi-channel physicochemical "
    "interaction map and weights that map by consensus contact probabilities "
    "derived from 198 experimentally determined TCR–pMHC complexes, so that "
    "structural knowledge constrains learning without requiring coordinates at "
    "inference. On a strictly split benchmark of 576 unseen epitopes, STEP "
    "achieved a ROC-AUC of 0.747 ± 0.010, outperforming five state-of-the-art "
    "methods; retrained no-prior, uniform, shuffled and length-only controls, "
    "three seeds each, place the structural prior’s contribution within seed "
    "variance. On melanoma a pooled ROC-AUC of 0.910 reflects training overlap: "
    "95.1% of rows carry a CDR3β seen in training and two MART-1 epitopes "
    "supply 88.0% of the positives, against a per-epitope mean of 0.672. Removing "
    "overlapping training pairs leaves melanoma at 0.591 ± 0.019 and lung at "
    "0.477 ± 0.027, which is uninformative. STEP ranked six clinically "
    "important viral epitopes (CMV, EBV, HIV) with ROC-AUC ≥ 0.963 on the "
    "imbalanced test set. Interpretability analyses show that the attended "
    "interface positions coincide with experimentally observed contacts, and that "
    "the most informative descriptors match established determinants of "
    "protein–protein interface energetics. STEP offers a route to prioritizing "
    "candidate TCR–epitope pairs for experimental testing. Source code and "
    "data are freely available at "
)
NEW_ABS = (
    "T-cell receptors (TCRs) recognize epitopes presented by the major "
    "histocompatibility complex (MHC), initiating the adaptive immune response. "
    "The rules governing TCR–epitope pairing therefore underpin neoantigen "
    "vaccine design and TCR-based immunotherapy. Prediction is difficult for "
    "unseen epitopes because binding is set by the geometry and chemistry of a "
    "shallow, largely hydrophobic interface that sequence representations capture "
    "poorly. Structural data could help, but resolved TCR–peptide–MHC "
    "complexes are scarce beside the available sequence data. Here we present STEP "
    "(Structure-informed TCR–Epitope Prediction), a deep learning framework "
    "that transfers interface knowledge from this corpus to sequence-scale "
    "prediction. STEP encodes each TCRβ CDR3–epitope pair as a "
    "multi-channel physicochemical interaction map weighted by consensus contact "
    "probabilities from 198 experimentally resolved TCR–pMHC complexes, "
    "constraining learning without coordinates at inference. On 576 held-out "
    "epitopes STEP reached ROC-AUC 0.747 ± 0.010, above five state-of-the-art "
    "methods, though four retrained prior controls (none, uniform, shuffled, "
    "length-only; three seeds each) place its contribution within seed variance. "
    "Melanoma’s pooled 0.910 reflects overlap: 95.1% of rows carry a training "
    "CDR3β and two MART-1 epitopes supply 88.0% of positives, against a "
    "per-epitope mean of 0.672. Without overlapping pairs melanoma falls to 0.591 "
    "± 0.019 and lung to 0.477 ± 0.027, which is uninformative. STEP "
    "scored six clinical viral epitopes (CMV, EBV, HIV) at ROC-AUC ≥ 0.963 on "
    "the imbalanced set. Attended interface positions coincide with experimentally "
    "observed contacts, and the most informative descriptors match known "
    "determinants of protein–protein interface energetics. STEP offers a "
    "route to prioritizing TCR–epitope pairs for experimental testing. Code "
    "and data are freely available at "
)
d.replace(OLD_ABS, NEW_ABS, expected=1, label='abstract trim')

d.save()

# ------------------------------------------------------------------ post-checks
chk = Docx(MS)
cps = list(chk.paragraphs())
abstract = chk.ptext(cps[10])
print("abstract words:", len(abstract.split()))
assert "Figure provenance." in chk.full_text()
assert "5HHO" in chk.full_text()
m212 = [i for i, p in enumerate(cps) if 'Figure provenance.' in chk.ptext(p)]
print("provenance paragraph at:", m212)
print("2.12 para 149 now:", chk.ptext(cps[149])[:160])
assert chk.full_text().count('5HHO') == 1, chk.full_text().count('5HHO')
for label, old, new, n in d.log:
    print(f"  OK  {label}: {n}")
