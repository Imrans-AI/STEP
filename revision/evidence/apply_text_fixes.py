"""
Apply the TEXT_FIXES_INVENTORY.md fixes to working copies of the two submitted files.

Sources in revision/as_submitted/ are never opened for writing; this script edits
revision/working/*_REVISED.docx only, and refuses to run if those copies are missing.

Every edit asserts its expected occurrence count, so a quote that has drifted stops
the run instead of silently doing nothing. The structured log is written to
revision/evidence/text_fixes_applied.json and rendered as diffs in
revision/evidence/TEXT_FIXES_APPLIED.md.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir('/home/muhammad-imran/PhD_data/TCR_Module_1')

from docx_tools import Docx                                      # noqa: E402

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
SUPP = 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'
for f in (MS, SUPP):
    assert os.path.exists(f), f"missing working copy {f}"

ms = Docx(MS)
sp = Docx(SUPP)
applied = []        # (id, file, section, old, new)
skipped = []        # (id, reason)


def E(cid, doc, section, old, new, n=1):
    doc.replace(old, new, expected=n, label=cid)
    applied.append(dict(id=cid, file='manuscript' if doc is ms else 'supplementary',
                        section=section, old=old, new=new, count=n))


def S(cid, reason):
    skipped.append(dict(id=cid, reason=reason))


# =====================================================================
# §4.2 — the ten equation delimiter objects, in one pass
# =====================================================================
want = []
for d in ms.delimiters():
    if (d['beg'], d['end']) == ('(', ')'):
        continue
    if d['beg'] == '{':
        want.append((d, '', 'eq. (1) contact matrix, piecewise: no closing brace'))
    elif d['beg'] == '|':
        tgt = '2.11, mean |SHAP|' if 'SHAP' in d['inner'] else '2.3.1, eq. (5) absolute value'
        want.append((d, '|', tgt))
    elif d['beg'] == '[':
        want.append((d, ']', f"interval/bracket: {d['inner'][:28]!r}"))
assert len(want) == 10, f"expected 10 mismatched delimiters, found {len(want)}"
for d, newend, where in want:
    before = ms.set_end_char(d, newend, label='R2-L4/L5/R3-6a §4.2')
    applied.append(dict(id='R2-L4 / R2-L5 / R3-6a (§4.2)', file='manuscript',
                        section=where,
                        old=f"{d['beg']} … {before}", new=f"{d['beg']} … {newend or '(none)'}",
                        count=1))

# =====================================================================
# §1 — Reviewer 3
# =====================================================================
S('R3-1a', 'Table S10 transposition is a layout change, not a text replacement. '
           'No definite replacement string in the inventory.')
S('R3-1b', 'Moving the "Training / inference command" column out of the table is a '
           'layout change. The commands themselves are unchanged.')

E('R3-2 / R2-C8', sp, 'Table S10, STEP row, "Reference" cell',
  'Submitted to Cell Systems', 'This work', n=2)
# the two cells differ; restore the provenance one to its fuller wording
sp.replace_in_paragraph(421, 'This work',
                        'This work; trained on the 600-epitope split, evaluated on the '
                        'shared balanced held-out test',
                        label='R3-2 / R2-C8 (provenance cell)')
applied.append(dict(id='R3-2 / R2-C8', file='supplementary',
                    section='Table S10, STEP row, "Provenance & reproduction status" cell',
                    old='Submitted to Cell Systems',
                    new='This work; trained on the 600-epitope split, evaluated on the '
                        'shared balanced held-out test', count=1))

S('R3-3', 'Caption says 225 epitopes, main text says 340. PROVENANCE_AUDIT.md shows the '
          'figure was built from a non-canonical checkpoint on the imbalanced test set, so '
          'which object is wrong cannot be settled from text. [[PENDING]] on the figure.')

E('R3-4a', ms, 'Methods 2.7',
  'Clinical viral epitopes (n = 6: four CMV pp65 variants, one EBV EBNA3C, one HIV gag) '
  'were analyzed separately. Statistical comparison between clinical and non-clinical '
  'epitope performance used the Mann–Whitney U test',
  'Six clinical viral epitopes (four CMV pp65 variants, one EBV EBNA3C, one HIV gag) '
  'were analyzed separately. Five of the six meet the per-epitope criteria on the '
  'balanced test set; all six do so on the imbalanced set. Statistical comparison '
  'between clinical and non-clinical epitope performance used the Mann–Whitney U test')
E('R3-4a (denominator)', ms, 'Methods 2.7',
  'appropriate for the small clinical group size (n = 5).',
  'appropriate for the small clinical group size (n = 5 on the balanced set).')

E('R3-4b', ms, 'Results, per-epitope performance',
  'The five clinical viral epitopes evaluable in the balanced test each achieved',
  'The five of six clinical viral epitopes evaluable in the balanced test (the sixth, '
  'NMVPMVATV, has seven balanced samples but no negative example, so its ROC-AUC is '
  'undefined there) each achieved')

E('R3-4d', sp, 'Table S4 caption',
  'All six epitopes achieve near-perfect discrimination (ROC-AUC ≥ 0.963).',
  'All six epitopes achieve near-perfect discrimination (ROC-AUC ≥ 0.963). Five of the '
  'six (NCVPMVATV, LLVPMVATV, VLVPMVATV, QASGNHAAGILTM, FLKEQGGL) also meet the '
  '≥ 5-sample, both-classes criterion on the balanced test set; NMVPMVATV does not, '
  'because all seven of its balanced samples are positive, which is why the balanced '
  'analysis reports five clinical epitopes and this table six.')

E('R3-5', ms, 'Methods 2.5, final paragraph',
  'For the retrained methods (STEP and DAISY) the partition is strictly epitope-disjoint. '
  'For the retrained methods the partition is strictly epitope-disjoint',
  'For the retrained methods the partition is strictly epitope-disjoint')

E('R3-6b', ms, 'Figure 5 caption, panel b', '(PPRO-95, VAL-96, THR-97)',
  '(PRO-95, VAL-96, THR-97)')

E('R3-6c(i)', ms, 'Methods 2.2, prose before eq. (2)',
  'The consensus contact probability map  was computed by averaging across all 198 structures:',
  'The raw contact-frequency map  was computed by averaging across all 198 structures:')
ms.set_math_text(32, 'W', 'Ẃ', label='R3-6c(i) symbol')
applied.append(dict(id='R3-6c(i)', file='manuscript',
                    section='Methods 2.2, inline symbol before eq. (2)',
                    old='W ∈ R20×15', new='Ẃ ∈ R20×15 (matches eq. (2), which defines Ẃ)',
                    count=1))
E('R3-6c(iii)', ms, 'Methods 2.2, prose before eq. (3)',
  'A small baseline was added to ensure non-zero weights at all positions, and the map '
  'was normalized to ',
  'A small baseline was added to ensure non-zero weights at all positions, and the raw '
  'map was then rescaled to give the consensus contact-probability map of eq. (3), on ')

# the appositive goes at the end of the sentence, not between the layer and the
# "for the melanoma … test sets" it governs
E('R3-7', ms, 'Methods 2.10',
  'we extracted embeddings from the penultimate convolutional layer (Dropout2d, '
  '128 × 20 × 15 = 38,400 dimensions) for the melanoma (n = 1,582) and lung cancer '
  '(n = 170) test sets.',
  'we extracted embeddings from the dropout layer of the DANet head (Dropout2d; '
  '128 × 20 × 15 = 38,400 dimensions) for the melanoma (n = 1,582) and lung cancer '
  '(n = 170) test sets, that is, after the head reduces the 512-channel ResNet output '
  'four-fold.')

S('R3-8', 'Data and Software Availability depends on three external checks (repository '
          'public at the stated URL, DOI resolving to the described deposit, deposited '
          'checkpoint being the one behind the reported numbers). [[PENDING]].')
S('R3-9', 'Reference 25 is uncited. Citing it at the ERGO-AE bullet requires inserting a '
          'superscript citation run and re-checking numbering; deleting it requires '
          'renumbering 26–57. Both are definite actions but neither is a text '
          'replacement the inventory specifies, and the choice is the authors\'. Flagged.')

# =====================================================================
# §2 — Reviewer 2, contradictions
# =====================================================================
E('R2-C1 (pool)', ms, 'Methods 2.1, Filtering and Quality Control',
  'After cross-database merging and deduplication of identical CDR3β–epitope pairs, '
  '99,323 unique positive pairs spanning 1,181 unique epitopes were retained.',
  'After cross-database merging and deduplication of identical CDR3β–epitope pairs, '
  '99,643 unique positive pairs spanning 1,281 unique epitopes were retained.')
E('R2-C1 (split)', ms, 'Methods 2.1, Data Splitting',
  'The epitope-level split allocated 600 epitopes to training and 581 to test; the '
  'remaining 19 epitopes had insufficient samples for reliable test evaluation.',
  'The epitope-level split allocated 600 epitopes to training, 100 to validation and 581 '
  'to test, accounting for all 1,281 epitopes in the pool.')
E('R2-C1 (validation set defined)', ms, 'Methods 2.1, Data Splitting',
  'The split yielded: a training set of 131,974 samples; an imbalanced test set of '
  '29,948 samples',
  'The split yielded: a training set of 131,974 samples; a validation set of 5,243 '
  'samples (100 epitopes, 320 positive pairs, 6.1% positive), epitope-disjoint from both '
  'the training and the test sets, whose role in model selection is described in '
  'Section 2.4 [[PENDING: R2-2]]; an imbalanced test set of 29,948 samples')

E('R2-C2', ms, 'Figure 1 caption, panel d',
  'Learned interaction features and global biophysical descriptors are concatenated and '
  'passed to the classifier',
  'Learned interaction features and global biophysical descriptors are fused by '
  'element-wise addition and passed to the classifier')

E('R2-C3', ms, 'Methods 2.1, Biological negatives',
  'indicating that STEP\'s performance is robust to the choice of negative-sampling '
  'strategy.',
  'indicating that STEP\'s performance is unchanged by this particular substitution; see '
  'Discussion for why harder negatives may still reduce it.')

E('R2-C4 (Fig 5a,b body)', ms, 'Results, structural visualization',
  'CDR3β positions PRO-95, VAL-96 and GLY-108 (Fig. 5a,b)',
  'CDR3β positions PRO-95, VAL-96 and THR-97 (Fig. 5a,b)')
E('R2-C4 (Fig 5c,d body)', ms, 'Results, structural visualization',
  'attention spanned epitope positions ILE-5, GLY-6 and VAL-10 and CDR3β positions '
  'THR-96, GLY-106 and GLY-108 (Fig. 5c,d)',
  'attention spanned epitope positions [[PENDING: R2-C4 — ALA-2, ALA-3, VAL-10 if panel c '
  'keeps the CAM it currently plots; VAL-10, GLY-6, ILE-5 if it is rebuilt from the 4QOK '
  'CAM]] and CDR3β positions [[PENDING: R2-C4 — THR-96, GLY-97, GLU-95 as plotted; '
  'GLY-97, GLU-95, GLU-102 if rebuilt]] (Fig. 5c,d)')
S('R2-C5', 'Table 1\'s DAISY PR-AUC of 0.658 is the correct full-test-set value '
           '(TEXT_FIXES_INVENTORY §8). The 0.663 exists only inside Fig 3c/3d, which were '
           'drawn from a 3,276-row dump. No manuscript text is wrong; the figure must be '
           'regenerated. No text edit made.')

E('R2-C6 (caption)', ms, 'Figure 6 caption, panel d',
  'Distribution of STEP binding scores across 9,127 HLA-A*02:01 IEDB nonamers; known '
  'MART-1 binders rank in the extreme right tail',
  'Distribution of STEP binding scores across 9,127 HLA-A*02:01 IEDB nonamers together '
  'with three 10-mer MART-1 peptides (9,130 scored peptides in total); percentiles are '
  'taken over all 9,130. Known MART-1 binders rank in the extreme right tail')
E('R2-C6 (body)', ms, 'Results, structure-informed interpretation',
  'we scored the fixed MART-1 TCR against 9,127 HLA-A*02:01 nonamers from IEDB:',
  'we scored the fixed MART-1 TCR against 9,127 HLA-A*02:01 nonamers from IEDB together '
  'with three 10-mer MART-1 peptides (9,130 peptides in total; the two highest-ranked '
  'names below, EAAGIGILTV and LLLGIGILVL, are 10-mers and AAGIGILTV is a nonamer):')

E('R2-C7', ms, 'Methods 2.10',
  'Dimensionality reduction was performed using Uniform Manifold Approximation and '
  'Projection (UMAP) with default parameters (n_neighbors = 15, min_dist = 0.3, '
  'random_state = 42)',
  # the closing parenthesis is kept: the reference superscript follows it, and
  # without it the citation number runs straight on from "42"
  'Dimensionality reduction was performed using Uniform Manifold Approximation and '
  'Projection (UMAP) with parameters (n_neighbors = 15, the library default; '
  'min_dist = 0.3; random_state = 42)')

# =====================================================================
# §3 — Reviewer 2, over-interpretation
# =====================================================================
E('R2-O1', ms, 'Abstract',
  'and discriminated clinically important viral epitopes with near-perfect accuracy.',
  'and ranked six clinically important viral epitopes (CMV, EBV, HIV) with ROC-AUC '
  '≥ 0.963 on the imbalanced test set.')
E('R2-O2', ms, 'Abstract, final sentence',
  'STEP provides a practical basis for neoantigen screening and TCR-based immunotherapy '
  'design.',
  'STEP offers a route to prioritizing candidate TCR–epitope pairs for experimental '
  'testing.')
E('R2-O3(a)', ms, 'Results, per-epitope performance',
  'each achieved ROC-AUC = 1.000 (n = 5), significantly higher than',
  'each achieved ROC-AUC = 1.000 (5 epitopes, 45–58 samples each on the imbalanced set; '
  'on the balanced set each contributes 6–11 samples, so a value of 1.000 states that '
  'every positive outranked every negative in a small sample), significantly higher than')
E('R2-O3(b)', ms, 'Discussion, paragraph 2',
  'STEP assigned perfect discrimination (ROC-AUC = 1.000) to the five clinical viral '
  'epitopes evaluable in the balanced test',
  'STEP ranked every positive above every negative (ROC-AUC = 1.000) for the five '
  'clinical viral epitopes evaluable in the balanced test, each of which contributes '
  'fewer than a dozen balanced samples')
E('R2-O3(c)', ms, 'Results, cancer',
  'with perfect discrimination for GARGVGKSAL and RFYKTLRAEQASQ (Supplementary Table S5).',
  'with ROC-AUC = 1.000 for GARGVGKSAL (2 positives of 7 samples) and RFYKTLRAEQASQ '
  '(2 positives of 8 samples); with two positives, the estimate has no resolution below '
  '1/2 (Supplementary Table S5).')
E('R2-O3(d,e)', sp, 'Table S5 caption',
  'Table S5. Per-epitope performance on melanoma cancer antigens. Epitopes sorted by '
  'ROC-AUC. Known melanoma antigens (MART-1, gp100) are annotated.',
  'Table S5. Per-epitope performance on melanoma cancer antigens. Epitopes sorted by '
  'ROC-AUC. Known melanoma antigens (MART-1, gp100) are annotated. ROC-AUC is reported '
  'with the positive count because several epitopes carry very few positives: a value of '
  '1.000 on 3 positives of 30 samples (EVDPIGHLY) states only that all three outranked '
  'every negative, and has no resolution below 1/3.')
S('R2-O4', 'Reference 55 (ESMFold) is cited for single-cell immune profiling, which it is '
           'not. Correcting it requires naming a single-cell immune-profiling reference; '
           'inventing a citation is not acceptable and choosing one is the authors\' call. '
           'Flagged for an author decision.')
E('R2-O5', ms, 'Reference 28', 'Adv. Sci. 2025, 13 (4), e12544.',
  'Adv. Sci. 2025, 12 (4), e12544.')

# =====================================================================
# §4 — Reviewer 2, language
# =====================================================================
E('R2-L1', ms, 'Results, baselines paragraph',
  'that lack structural priors Precision–recall', 'that lack structural priors. Precision–recall')
E('R2-L2', ms, 'Results, same paragraph',
  'achieving the highest AUPR among all six models (Fig. 3d)..',
  'achieving the highest AUPR among all six models (Fig. 3d).')
E('R2-L3 (coloured)', ms, 'Methods 2.10, Fig 3 caption g/h/i', 'coloured', 'colored', n=4)
E('R2-L3 (colour)', ms, 'Figure 4 caption', 'point colour indicates', 'point color indicates')
E('R2-L3 (tumour)', ms, 'Abstract, Methods 2.2, Results ×2', 'tumour', 'tumor', n=4)
E('R2-L3 (behaviour)', ms, 'Results, Discussion', 'behaviour', 'behavior', n=2)
E('R2-L3 (tumour, supp)', sp, 'Fig S2 caption', 'tumour', 'tumor')

# =====================================================================
# §6 — additional inconsistencies
# =====================================================================
E('X1 (Table S10 caption)', sp, 'Table S10 caption',
  'STEP, DAISY, and TEINet were retrained on the STEP 600-epitope training split '
  '(131,974 pairs); ERGO-LSTM, ERGO-AE, and ATM-TCR were evaluated zero-shot using the '
  'authors\' released weights and were not trained on our data',
  'STEP and DAISY were retrained on the STEP 600-epitope training split (131,974 pairs); '
  'ERGO-LSTM, ERGO-AE and TEINet were evaluated using the authors\' released weights and '
  'were not trained on our data; ATM-TCR was trained with its own default pipeline on its '
  'bundled dataset rather than on our split')
E('X1 (TEINet provenance cell)', sp, 'Table S10, TEINet row',
  'Provenance: retrained by us on our 600-epitope training split '
  '(transformer_20260318_135745). Evaluated on our held-out test.',
  'Provenance: authors\' released weights (teinet_data.pth); not retrained on our data. '
  'Evaluated on our held-out test.')
E('X1 (ATM-TCR provenance cell)', sp, 'Table S10, ATM-TCR row',
  'Provenance: authors\' released weights (no retrain found); evaluated zero-shot on our '
  'held-out test',
  'Provenance: trained with its own default pipeline on its bundled dataset '
  '(combined_dataset.csv), not on our split; evaluated on our held-out test')

E('X2', ms, 'Methods 2.3.3',
  'a ResNet-style convolutional neural network41 consisting of four residual blocks.',
  'a ResNet-style convolutional neural network41 consisting of a 5→64 stem convolution '
  'followed by four residual stages of two blocks each (eight residual blocks in total).')
E('X3', ms, 'Methods 2.12, final sentence',
  '(PDB: 3GSN for NLVPMVATV/CMV; 4QOK for EAAGIGILTV/MART-1 by mapping position-averaged '
  'attention scores onto residue B-factors and rendering with PyMOL',
  '(PDB 3GSN for NLVPMVATV/CMV; 4QOK for EAAGIGILTV/MART-1), by mapping position-averaged '
  'attention scores onto residue B-factors and rendering with PyMOL')
E('X4', ms, 'Methods 2.3',
  'takes two inputs a TCRβ CDR3 sequence and a peptide epitope sequence and outputs',
  'takes two inputs, a TCRβ CDR3 sequence and a peptide epitope sequence, and outputs')
S('X5', 'The text-box anchor residue "center635"/"069850" is in neither package: searched '
        'every .xml part of both files, zero occurrences. Nothing to remove.')

# =====================================================================
ms.save()
sp.save()
json.dump(dict(applied=applied, skipped=skipped),
          open('revision/evidence/text_fixes_applied.json', 'w'), indent=2)
print(f"applied {len(applied)} edits, skipped/flagged {len(skipped)}")
for a in applied:
    print(f"  [{a['id']}] {a['file']} — {a['section']}")
for s_ in skipped:
    print(f"  SKIPPED [{s_['id']}]")
