# -*- coding: utf-8 -*-
"""
Three additions to revision/working/Manuscript_fv_1_REVISED.docx.

1. Methods 2.2 rewritten to describe the corrected construction (threshold on the
   valid residue block before padding; per-cell coverage as the denominator;
   normalisation over covered cells), plus a new paragraph recording what changed
   from the released code and defining "the corrected pipeline" at its first use
   in the Methods.
2. A Results subsection reporting the similarity-disjoint evaluation that R2-1
   asked for, with a new Table 3.  It is placed after the ablation subsection so
   that the new table is Table 3 and nothing has to be renumbered.
3. One Discussion sentence placing 0.747 and 0.660 as measurements under two split
   criteria rather than as competing headlines.

Numbers come from evidence/R1-2_padding_audit.md, evidence/R1-2_prior_v2.md and
evidence/R2-1_similarity_split.md (Parts A2, B1, B3, B4, C2, C4).
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


def one_run(p, text):
    """Reduce a cloned paragraph to a single run carrying `text`."""
    runs = list(p.iter(W + 'r'))
    for r in runs[1:]:
        r.getparent().remove(r)
    t = runs[0].find(W + 't')
    t.text = text
    t.set(XS, 'preserve')
    return p


def clone_para(src, text):
    return one_run(copy.deepcopy(src), text)


# ===================================================== 1. Methods 2.2 ==========
d.replace_in_paragraph(
    33,
    "All contact matrices were zero-padded to a fixed size of ",
    "The 6.0 Å threshold was applied to the valid residue block of each distance "
    "matrix — the first h rows and w columns, for a structure with h CDR3β and "
    "w epitope residues — and each resulting binary contact matrix was then "
    "zero-padded to a fixed size of ",
    label='2.2 threshold before padding')
d.replace_in_paragraph(
    33,
    " (maximum CDR3β length × maximum epitope length). The raw contact-frequency "
    "map  was computed by averaging across all 198 structures:",
    " (maximum CDR3β length × maximum epitope length), so that padded cells carry "
    "no contact. The raw contact-frequency map  was computed cell by cell, each cell "
    "averaged only over the structures whose residues reach it:",
    label='2.2 coverage averaging')

# The inline "N = 198" becomes the per-cell coverage count.
d.set_math_text(35, '198', 'cᵢⱼ', label='2.2 denominator is per-cell coverage')
d.replace_in_paragraph(
    35,
    ". A small baseline was added to ensure non-zero weights at all positions, and the "
    "raw map was then rescaled to give the consensus contact-probability map of eq. (3), "
    "on ",
    ", the number of structures whose CDR3β and epitope residues both reach cell "
    "(i, j); it is at least 20 for every cell that carries a contact, and is saved "
    "alongside the map. A small baseline was added to ensure non-zero weights at all "
    "positions, and the raw map was then rescaled over covered cells only to give the "
    "consensus contact-probability map of eq. (3), on ",
    label='2.2 coverage definition and rescaling')

CORRECTION = (
    " The code released with the submitted manuscript applied these two operations in "
    "the opposite order: each distance matrix was zero-padded to 20 × 15 first and "
    "the 6.0 Å threshold applied second, so every padded cell held a distance of "
    "zero, passed the threshold, and was counted as a contact in all 198 structures, "
    "each cell being divided by 198 irrespective of how many structures reached it. The "
    "effect was not an inversion of the prior but a saturation of it. Restricted to "
    "cells that are never padded, the highest weights already fell on the interface, "
    "peaking at 0.389 at CDR3β position 5 × epitope position 6 against a floor "
    "of 0.091, while padding-derived mass reached 1.0 and held the normalisation "
    "denominator at 1.1, compressing the true-contact signal into the lower part of the "
    "weight range. Rebuilding the map as described above lowers that denominator to "
    "0.428, moves the peak from a padded cell to the interface cell, reduces the number "
    "of cells saturated at 1.0 from 82 to one, and raises the enrichment of the contact "
    "core over background from 0.387-fold — that is, inverted — to 2.325-fold. "
    "Masking the padded positions of the physicochemical map was also implemented and "
    "verified to be a no-op on these data, since those positions are already exactly "
    "zero; the measurable correction is the prior itself. Throughout this paper, the "
    "corrected pipeline denotes this rebuilt prior together with the whitespace-"
    "normalised splits of Section 2.1 and the models retrained on them with three seeds. "
    "Quantities that predate these corrections are identified as such wherever they are "
    "reported."
)
head = clone_para(ps[22], 'Correction to the released construction.')   # bold run-in
runs = list(head.iter(W + 'r'))
body_run = copy.deepcopy(list(ps[22].iter(W + 'r'))[1])
bt = body_run.find(W + 't')
bt.text = CORRECTION
bt.set(XS, 'preserve')
runs[0].addnext(body_run)
ps[37].addnext(head)
d.log.append(('2.2 correction paragraph', '', CORRECTION[:40], 1))


# ============================================ 2. Results: similarity split =====
HEAD = "Performance under a similarity-disjoint split"
P1 = (
    "Epitope-level disjointness was enforced by exact string matching, which does not "
    "make a test epitope unfamiliar. Measuring the Levenshtein distance from every test "
    "epitope to its nearest training epitope, 183 of the 576 test epitopes (31.8%) are a "
    "single substitution away from a training epitope and 200 (34.7%) are at 80% "
    "identity or above. Per-epitope performance is concentrated in that neighbourhood "
    "(Table 3): on epitopes one substitution from training, STEP reaches 0.909 against "
    "DAISY's 0.730, whereas on epitopes at distance four or more — 202 of the 342 "
    "evaluable epitopes, carrying 1,559 of the 2,558 evaluable rows — the two are "
    "level at 0.671 and 0.666. All seven clinical viral epitopes in the test set lie at "
    "distance one, and all of them belong to a single 23-member mutational cluster around "
    "NLVPMVATV that falls on both sides of the split, so the ROC-AUC of 1.000 reported "
    "for them above is recall of point mutants of training epitopes rather than "
    "recognition of unfamiliar antigens."
)
P2 = (
    "We therefore rebuilt the split so that near neighbours cannot straddle it. The 1,276 "
    "pool epitopes were clustered by single linkage, two epitopes being linked if their "
    "edit distance is at most two or their sequence identity at least 80%, giving 824 "
    "clusters: 655 singletons, the largest holding 85 epitopes, with the five largest all "
    "mutational ladders around one wild-type sequence. Whole clusters were then assigned "
    "to training, validation or test, preserving the original epitope budget as closely "
    "as indivisible clusters allow: 601 training, 101 validation and 574 test epitopes, "
    "over 125,061, 6,044 and 33,892 rows, with a balanced test set of 7,648 samples "
    "spanning 540 clusters. No cluster crosses a boundary at the epitope, cluster or "
    "pair level; the minimum edit distance from any test epitope to any training epitope "
    "is three and the maximum identity 0.750. Training-set size is within 4% of the "
    "epitope-disjoint split, so the comparison below is not confounded by a large change "
    "in the amount of training data."
)
P3 = (
    "STEP was retrained from scratch on this split with the corrected prior and three "
    "seeds, and DAISY was retrained on the positives of the same training split with its "
    "authors' own procedure. STEP reaches ROC-AUC 0.660 ± 0.020 and PR-AUC "
    "0.620 ± 0.011; DAISY reaches 0.655 and 0.600. The difference of 0.011 between "
    "the three-seed ensembles is not significant under a correlated DeLong test "
    "(P = 0.148) nor under a bootstrap that resamples the 540 test clusters rather than "
    "the rows (P = 0.427), and the cluster-bootstrap intervals for the two AUCs overlap "
    "([0.639, 0.691] and [0.640, 0.670]). The seed-to-seed range of STEP on this split, "
    "0.640 to 0.679, is larger than its margin over DAISY, and one of the three seeds "
    "falls behind DAISY, so no ranking between the two methods is supported here. Both "
    "models lose accuracy when near neighbours are removed, STEP roughly twice as much as "
    "DAISY (−0.088 against −0.047 in ROC-AUC), which is the quantitative form of "
    "the same observation."
)
CAP = (
    "Table 3. Per-epitope ROC-AUC on the balanced unseen-epitope test set, stratified by "
    "the Levenshtein distance from each test epitope to its nearest training epitope. "
    "Values are mean ± SD across the epitopes in each stratum, over the 342 epitopes "
    "with at least five test samples and both classes present. STEP is shown for the "
    "pre-specified single seed and for the three-seed ensemble."
)
ROWS = [
    ("Distance to nearest\ntraining epitope", "Epitopes", "Test rows",
     "STEP (seed 42)", "STEP (3-seed ensemble)", "DAISY"),
    ("1", "111", "781", "0.896 ± 0.199", "0.909 ± 0.189", "0.730 ± 0.233"),
    ("2", "16", "109", "0.818 ± 0.214", "0.872 ± 0.197", "0.711 ± 0.267"),
    ("3", "13", "109", "0.720 ± 0.331", "0.674 ± 0.361", "0.664 ± 0.266"),
    ("≥ 4", "202", "1,559", "0.639 ± 0.270", "0.671 ± 0.259",
     "0.666 ± 0.245"),
]
WIDTHS = [2400, 1100, 1100, 1580, 1780, 1179]          # sums to 9139, as Table 2's grid

body = d.tree.find(W + 'body')
kids = list(body)
tbl2 = list(d.tree.iter(W + 'tbl'))[1]
anchor = kids[kids.index(tbl2) + 1]                    # the empty paragraph after Table 2

new_tbl = copy.deepcopy(tbl2)
grid = new_tbl.find(W + 'tblGrid')
cols = grid.findall(W + 'gridCol')
for c in cols[6:]:
    grid.remove(c)
for c, wdt in zip(grid.findall(W + 'gridCol'), WIDTHS):
    c.set(W + 'w', str(wdt))
trs = new_tbl.findall(W + 'tr')
assert len(trs) == len(ROWS), (len(trs), len(ROWS))
for tr, vals in zip(trs, ROWS):
    tcs = tr.findall(W + 'tc')
    for tc in tcs[6:]:
        tr.remove(tc)
    tcs = tr.findall(W + 'tc')
    for tc, wdt, text in zip(tcs, WIDTHS, vals):
        tcPr = tc.find(W + 'tcPr')
        if tcPr is not None:
            tcW = tcPr.find(W + 'tcW')
            if tcW is not None:
                tcW.set(W + 'w', str(wdt))
        cps = tc.findall(W + 'p')
        for extra in cps[1:]:
            tc.remove(extra)
        one_run(cps[0], text.replace('\n', ' '))

blank = copy.deepcopy(anchor)
for node in (blank, new_tbl, clone_para(ps[182], CAP),
             clone_para(ps[153], P3), clone_para(ps[153], P2),
             clone_para(ps[153], P1), clone_para(ps[186], HEAD)):
    anchor.addnext(node)
d.log.append(('Results: similarity-disjoint subsection + Table 3', '', HEAD, 1))

# ================================================ 3. Discussion sentence =======
# Paragraph indices have shifted: six paragraphs and a table were inserted above.
# Locate the Discussion paragraph by content rather than by its pre-insertion index.
disc = d.find("Its source, however, is not established")
assert len(disc) == 1, disc
d.replace_in_paragraph(
    disc[0],
    "Its source, however, is not established:",
    "The two figures this paper reports for unseen-epitope performance, 0.747 here and "
    "0.660 on a similarity-disjoint split, are measurements under two different "
    "definitions of what counts as unseen rather than competing estimates of one "
    "quantity: the first excludes test epitopes seen verbatim in training, the second "
    "excludes their near neighbours as well, and the gap between them is the price of "
    "the stricter criterion. Its source, however, is not established:",
    label='Discussion: 0.747 and 0.660 as two criteria')

d.save()
for label, old, new, n in d.log:
    print(f"  OK  {label}: {n}")

chk = Docx(MS)
cps = list(chk.paragraphs())
print("\nparagraphs:", len(cps), "tables:", len(list(chk.tree.iter(W + 'tbl'))))
full = chk.full_text()
for must in ('Correction to the released construction.', 'the corrected pipeline denotes',
             'Performance under a similarity-disjoint split', 'Table 3.',
             '0.660', '0.427', '31.8%', '23-member'):
    assert must in full, must
print("all assertions passed")
