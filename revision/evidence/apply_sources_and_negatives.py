# -*- coding: utf-8 -*-
"""
Four additions to revision/working/Manuscript_fv_1_REVISED.docx.

1. Methods 2.1 names UniPMT with its contribution, replacing the "6.7% not uniquely
   attributable" wording.
2. Methods 2.1 states that the submitted splits were disjoint only as raw strings,
   with the measured overlap and the remedy after it.
3. A Discussion limitation on negative-set construction.
4. The similarity-disjoint construction moves from Results into Methods 2.1; the
   Results subsection keeps the measurement and refers back.

Numbers: evidence/R2-1_synthetic_epitopes.md section 5, evidence/R2-1_whitespace_contamination.md
sections 2-3, evidence/R2-1_similarity_split.md Parts B and C4.
"""
import copy
import sys

sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx                                    # noqa: E402

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
XS = '{http://www.w3.org/XML/1998/namespace}space'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'

d = Docx(MS)
ps = list(d.paragraphs())


def run_in(src_idx, head, body):
    """Clone a bold-run-in Methods paragraph and set its two runs."""
    p = copy.deepcopy(ps[src_idx])
    runs = list(p.iter(W + 'r'))
    for r in runs[2:]:
        r.getparent().remove(r)
    for r, txt in zip(runs[:2], (head, body)):
        t = r.find(W + 't')
        t.text = txt
        t.set(XS, 'preserve')
    return p


# ------------------------------------------------- 1. the fourth source -------
d.replace_in_paragraph(
    27,
    "Training positives derive from three databases with highly uneven contributions: "
    "VDJdb (85.3% of pairs), McPAS-TCR (9.8%), and IEDB (2.0%), with 6.7% not uniquely "
    "attributable after sequence normalization (Table S11).",
    "Training positives derive from four sources rather than the three named in the "
    "submitted manuscript. The multi-source file from which the splits are built "
    "attributes its 99,643 positive pairs to VDJdb (82,617), UniPMT (8,256), McPAS-TCR "
    "(6,585) and IEDB (2,185) (Table S11). UniPMT contributes the second-smallest share "
    "of pairs but the largest number of distinct epitopes — 522 of the 1,281 in the "
    "pool, against 386 from VDJdb, 382 from IEDB and 291 from McPAS-TCR — and "
    "supplied 210 of the 581 epitopes in the submitted test set. Only its positive pairs "
    "were imported.",
    label='2.1 name UniPMT with its contribution')

# -------------------------------------- 2. disjointness held only on raw strings
NORM_HEAD = "Normalisation of the submitted splits."
NORM_BODY = (
    " The disjointness reported for the submitted splits holds only as a comparison of "
    "raw strings. Epitope and CDR3β fields were carried through the build chain "
    "without normalisation except on the VDJdb path, so non-breaking spaces, trailing "
    "whitespace and case variants survived into the split files. After normalising "
    "— stripping whitespace, removing U+00A0 and upper-casing — the submitted "
    "training and test sets share one epitope and four exact TCR–epitope pairs, and "
    "the validation and test sets share one of each, against the zero overlap reported. "
    "The leak is not a near match: for every affected pair the contaminated and the clean "
    "spelling produce byte-identical physicochemical maps, because the stray character is "
    "absent from the property tables and leaves its column at zero, so the network cannot "
    "distinguish the test pair from the training pair and only the index key differs. "
    "Three of the four training-to-test pairs are label-inconsistent as well, appearing "
    "as positives in training and as negatives in test. Every split used in this work was "
    "therefore rebuilt after normalisation, which merges four contaminated spellings and "
    "reduces the pool from 1,281 to 1,276 distinct epitopes; in the rebuilt splits "
    "epitope and exact-pair overlap between training, validation and test are zero under "
    "both the raw and the normalised comparison."
)

# ----------------------------------- 4. similarity-split construction to Methods
SIM_HEAD = "Similarity-disjoint split."
SIM_BODY = (
    " Exact-string disjointness does not make a test epitope unfamiliar, so a second "
    "split was built in which near neighbours cannot straddle the boundary, and is used "
    "for the robustness analysis reported in Section 3.7. The 1,276 pool epitopes were "
    "clustered by single linkage, two epitopes being linked if their Levenshtein distance "
    "is at most two or their sequence identity at least 80%, giving 824 clusters: 655 "
    "singletons, the largest holding 85 epitopes, with the five largest all mutational "
    "ladders around one wild-type sequence. Whole clusters were then assigned to "
    "training, validation or test, preserving the original epitope budget as closely as "
    "indivisible clusters allow: 601 training, 101 validation and 574 test epitopes, over "
    "125,061, 6,044 and 33,892 rows, with a balanced test set of 7,648 samples spanning "
    "540 clusters. No cluster crosses a boundary at the epitope, cluster or pair level; "
    "the minimum edit distance from any test epitope to any training epitope is three and "
    "the maximum identity 0.750. Training-set size is within 4% of the epitope-disjoint "
    "split, so the comparison is not confounded by a large change in the amount of "
    "training data."
)
ps[27].addnext(run_in(22, SIM_HEAD, SIM_BODY))        # inserted second, so lands last
ps[26].addnext(run_in(22, NORM_HEAD, NORM_BODY))
d.log.append(('2.1 normalisation statement', '', NORM_HEAD, 1))
d.log.append(('2.1 similarity-disjoint construction', '', SIM_HEAD, 1))

# ---------------------- Results: drop the moved paragraph, keep the antecedent --
sim_res = d.find("We therefore rebuilt the split so that near neighbours cannot straddle it.")
assert len(sim_res) == 1, sim_res
moved = list(d.paragraphs())[sim_res[0]]
moved.getparent().remove(moved)
d.log.append(('Results: construction paragraph removed (now Methods 2.1)', '', '', 1))

d.replace(
    "STEP was retrained from scratch on this split with the corrected prior and three "
    "seeds,",
    "The split was therefore rebuilt so that near neighbours cannot straddle it, by "
    "clustering the 1,276 pool epitopes at an edit distance of two or 80% identity and "
    "assigning whole clusters to each partition (Section 2.1). STEP was retrained from "
    "scratch on it with the corrected prior and three seeds,",
    expected=1, label='Results: restore the antecedent after the move')

# ------------------------------------- 3. Discussion: negative-set limitation ---
NEG = (
    "A further limitation concerns the construction of the negative set rather than the "
    "model. Every negative pair is formed by re-pairing a CDR3β that occurs as a "
    "positive elsewhere in the data, so a receptor absent from the training set is almost "
    "always a positive in the test set. The model-free rule “predict non-binding if "
    "the CDR3β did not occur in training” therefore reaches ROC-AUC 0.7788 on "
    "the balanced unseen-epitope test set and 0.8261 on the similarity-disjoint split "
    "— above every trained model evaluated here, including STEP. Any model able to "
    "represent receptor novelty can exploit this, and the absolute figures should be read "
    "with that in mind. The comparison between methods is less affected than the absolute "
    "values: restricted to the stratum in which the CDR3β was seen in training, "
    "which removes the shortcut, STEP reaches 0.7005 against DAISY’s 0.6664 "
    "(correlated DeLong P = 0.00067). Measuring absolute performance free of this effect "
    "would require negatives built from receptors never observed as binders, or from "
    "biologically motivated non-binders."
)
lim = d.find("Several limitations qualify these conclusions.")
assert len(lim) == 1, lim
limp = list(d.paragraphs())[lim[0]]
neg = copy.deepcopy(limp)
runs = list(neg.iter(W + 'r'))
for r in runs[1:]:
    r.getparent().remove(r)
t = runs[0].find(W + 't')
t.text = NEG
t.set(XS, 'preserve')
limp.addnext(neg)
d.log.append(('Discussion: negative-set limitation', '', NEG[:40], 1))

d.save()
for label, old, new, n in d.log:
    print(f"  OK  {label}: {n}")

chk = Docx(MS)
full = chk.full_text()
for must in ('UniPMT (8,256)', '522 of the 1,281', 'Normalisation of the submitted splits.',
             'four exact TCR–epitope pairs', 'Similarity-disjoint split.',
             '0.7788', '0.8261', '0.00067'):
    assert must in full, must
assert full.count("We therefore rebuilt the split") == 0
assert full.count('824 clusters') == 1, full.count('824 clusters')
print("all assertions passed")
