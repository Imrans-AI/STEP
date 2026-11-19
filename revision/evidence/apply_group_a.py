"""
Group A of the numbers pass, plus the B3-B6 results and the R1-7 resolution.

Sources: NUMBERS_PASS.md §1-§4, §8 and numbers_pass/b3456.json. Read-only on
revision/as_submitted/.
"""
import copy
import sys

sys.path.insert(0, 'revision/evidence')
from lxml import etree

from docx_tools import Docx, W, WT

WR, WRPR, WTC, WTR, WTBL = (f'{{{W}}}r', f'{{{W}}}rPr', f'{{{W}}}tc',
                            f'{{{W}}}tr', f'{{{W}}}tbl')
WPPR, WP, WGRID, WTCPR = (f'{{{W}}}pPr', f'{{{W}}}p', f'{{{W}}}tblGrid',
                          f'{{{W}}}tcPr')
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
SI = 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'
PM, EM, EN = '±', '—', '–'


def set_cell(tc, text):
    """Replace a table cell's visible text, keeping its first run's formatting."""
    ts = [t for p in tc.iter(WP) for t in p.iter(f'{{{W}}}t')]
    assert ts, 'cell has no text run'
    ts[0].text = text
    for t in ts[1:]:
        t.text = ''
    if text != text.strip():
        ts[0].set(XML_SPACE, 'preserve')


def cells(tr):
    return list(tr.iter(WTC))


# ======================================================================== MS
d = Docx(MS)
before = [d.ptext(p) for p in d.paragraphs()]


def para(needle):
    hits = [i for i, t in enumerate(before) if needle in t]
    assert len(hits) == 1, f'{len(hits)} paragraphs contain {needle[:55]!r}'
    return hits[0]


# ---------------------------------------------- A2: Table 1, one column only
tbl = list(d.tree.iter(WTBL))[0]
rows = list(tbl.iter(WTR))
assert len(rows) == 8 and len(cells(rows[1])) == 9, 'Table 1 shape changed'
grid = tbl.find(WGRID)
gcols = list(grid)
assert len(gcols) == 9

# row 0 carries five cells, each scenario spanning two grid columns; keep the
# label cell and the Unseen-Epitope cell, and widen the latter to the two
# surviving data columns.
r0 = cells(rows[0])
assert ' '.join(''.join(x.text or '' for x in r0[3].iter(f'{{{W}}}t')).split()) \
    == 'Unseen-Epitope'
for tc in (r0[1], r0[2], r0[4]):
    tc.getparent().remove(tc)
# rows 1-7: keep columns 0, 5, 6
for tr in rows[1:]:
    cs = cells(tr)
    assert len(cs) == 9
    for k, tc in enumerate(cs):
        if k not in (0, 5, 6):
            tc.getparent().remove(tc)
# grid: keep 0, 5, 6
for k, gc in enumerate(gcols):
    if k not in (0, 5, 6):
        grid.remove(gc)

VALUES = {  # model -> (ROC, PR) on the corrected unseen-epitope benchmark
    'ATM-TCR': ('0.585', '0.616'),
    'ERGO-AE': ('0.634', '0.641'),
    'ERGO-LSTM': ('0.583', '0.574'),
    'TEINet': ('0.524', '0.577'),
    'DAISY': ('0.701', '0.659'),
    'STEP': (f'0.747 {PM} 0.010', f'0.747 {PM} 0.011'),
}
for tr in rows[2:]:
    cs = cells(tr)
    name = ' '.join(''.join(x.text or '' for x in cs[0].iter(f'{{{W}}}t')).split())
    assert name in VALUES, name
    set_cell(cs[1], VALUES[name][0])
    set_cell(cs[2], VALUES[name][1])

d.replace(
    'Table 1. Performance comparison of STEP with five baseline methods across four '
    'generalization scenarios: Seen-Pair (regular cross-validation), Unseen-TCR '
    '(TCR-grouped), Unseen-Epitope (epitope-grouped, balanced), and Unseen-Pair '
    '(pair-grouped).',
    'Table 1. Performance comparison of STEP with five baseline methods on the '
    'unseen-epitope benchmark (epitope-grouped, balanced; n = 3,314 over 576 '
    f'epitopes). STEP is the mean {PM} SD over three training seeds; the five '
    'baselines are single fixed models. ATM-TCR and ERGO-LSTM differ by 0.002 '
    f'(95% CI −0.036 to +0.041) and are not ranked against each other. The '
    'Seen-Pair, Unseen-TCR and Unseen-Pair partitions were not reconstructed under '
    'the corrected pipeline (Section 2.5) and are no longer reported.',
    expected=1, label='A2 Table 1 caption')

# ---------------------------------------------- A1: Table 2
tbl2 = list(d.tree.iter(WTBL))[1]
r2 = list(tbl2.iter(WTR))
assert len(r2) == 5
T2 = {
    'STEP (Full)': (f'0.747 {PM} 0.010', f'0.747 {PM} 0.011', f'0.210 {PM} 0.198',
                    f'0.662 {PM} 0.019', f'0.680 {PM} 0.012', f'0.645 {PM} 0.043',
                    f'0.696 {PM} 0.035'),
    'STEP (No structure)': (f'0.754 {PM} 0.009', f'0.733 {PM} 0.010',
                            f'0.027 {PM} 0.010', f'0.708 {PM} 0.020',
                            f'0.653 {PM} 0.007', f'0.776 {PM} 0.057',
                            f'0.586 {PM} 0.042'),
}
RENAME = {'STEP (Bio negatives)': 'STEP (Bio negatives, original pipeline)',
          'STEP (300 epitopes)': 'STEP (300 epitopes, original pipeline)'}
for tr in r2[1:]:
    cs = cells(tr)
    name = ' '.join(''.join(x.text or '' for x in cs[0].iter(f'{{{W}}}t')).split())
    if name in T2:
        for k, v in enumerate(T2[name], start=1):
            set_cell(cs[k], v)
    elif name in RENAME:
        set_cell(cs[0], RENAME[name])
    else:
        raise AssertionError(f'unexpected Table 2 row {name!r}')

d.replace(
    'Table 2. Performance of STEP variants on the balanced unseen-epitope test set '
    '(n = 3,314). Threshold-dependent metrics (F1, Precision, Recall, Specificity) '
    "are computed at the Youden's J optimal threshold for each variant.",
    'Table 2. Performance of STEP variants on the balanced unseen-epitope test set '
    f'(n = 3,314). The full and no-structure variants are the mean {PM} SD over '
    'three training seeds retrained under the corrected pipeline; their '
    "threshold-dependent metrics are computed at the Youden's J threshold fitted on "
    'the validation set (Section 2.6). The biological-negative and 300-epitope '
    'variants were not regenerated (Section 2.9) and are reported as originally '
    'computed, at a threshold fitted on the test set.',
    expected=1, label='A1 Table 2 caption')

# ---------------------------------------------- A4 + A3 + B3-B6, prose
d.replace(
    'STEP achieved a ROC-AUC of 0.755 (95% CI 0.739' + EN + '0.771) and a PR-AUC of '
    '0.736 (Fig. 2, Table 1).',
    f'STEP achieved a ROC-AUC of 0.747 {PM} 0.010 and a PR-AUC of 0.747 {PM} 0.011 '
    f'(mean {PM} SD over three training seeds; Fig. 2, Table 1).',
    expected=1, label='A4 Results, headline')

d.replace('outperforming five state-of-the-art methods (all P < 0.001)',
          'outperforming five state-of-the-art methods',
          expected=1, label='A4 Abstract, drop the superseded P claim')
d.replace('STEP achieved a ROC-AUC of 0.755, outperforming',
          f'STEP achieved a ROC-AUC of 0.747 {PM} 0.010, outperforming',
          expected=1, label='A4 Abstract, headline')
d.replace('On a strictly split benchmark of 581 unseen epitopes',
          'On a strictly split benchmark of 576 unseen epitopes',
          expected=1, label='A3 Abstract, epitope count')
d.replace('evaluated it on a strictly split, epitope-disjoint benchmark of 581 '
          'unseen epitopes',
          'evaluated it on a strictly split, epitope-disjoint benchmark of 576 '
          'unseen epitopes',
          expected=1, label='A3 Introduction, epitope count')
d.replace('On a balanced benchmark of 581 unseen epitopes, STEP achieved ROC-AUC '
          '0.755, significantly exceeding all five tested baselines, including the '
          f'strongest, DAISY (0.701; P = 1.32 × 10⁻⁵)',
          'On a balanced benchmark of 576 unseen epitopes, STEP achieved ROC-AUC '
          f'0.747 {PM} 0.010, exceeding all five tested baselines, including the '
          'strongest, DAISY (0.701; epitope-level cluster bootstrap '
          'P = 2.80 × 10⁻³)',
          expected=1, label='A4+A3 Discussion, headline')
d.replace('a balanced, epitope-disjoint test set of 3,314 samples spanning 581 '
          'epitopes absent from training',
          'a balanced, epitope-disjoint test set of 3,314 samples spanning 576 '
          'epitopes absent from training',
          expected=1, label='A3 Results opening, epitope count')

d.replace(
    # straight apostrophe in this sentence, unlike the other Youden's mentions
    "STEP achieved a ROC-AUC of 0.755 with the Youden's J optimal threshold at "
    'τ = 0.027 (Table 1).',
    f'STEP achieved a ROC-AUC of 0.747 {PM} 0.010, at a validation-fitted '
    f"Youden's J threshold of τ = 0.210 {PM} 0.198 (Table 1).",
    expected=1, label='A4 Results, per-epitope paragraph headline')

# B5 — the clinical sentence: under v2 it is LLVPMVATV that is missing, not
# NMVPMVATV, and for a different reason.
d.replace(
    'The five of six clinical viral epitopes evaluable in the balanced test (the '
    'sixth, NMVPMVATV, has seven balanced samples but no negative example, so its '
    'ROC-AUC is undefined there) each achieved ROC-AUC = 1.000 (5 epitopes, '
    f'45{EN}58 samples each on the imbalanced set; on the balanced set each '
    'contributes 6' + EN + '11 samples, so a value of 1.000 states that every positive '
    'outranked every negative in a small sample)',
    'The five of six clinical viral epitopes evaluable in the balanced test (the '
    'sixth, LLVPMVATV, is absent from it) each achieved ROC-AUC = 1.000 on the '
    'three-seed ensemble, with each epitope contributing 7' + EN + '11 balanced samples, '
    'so a value of 1.000 states that every positive outranked every negative in a '
    'small sample; across individual seeds the only departure is NMVPMVATV at 0.929 '
    'for one seed',
    expected=1, label='B5 Results, clinical epitopes')

# B3 — calibration
d.replace(
    'In the seen-pair regime STEP is well-calibrated (expected calibration error '
    '[ECE] 0.026, Brier 0.039 at ROC 0.990). On unseen epitopes the network is '
    f'systematically under-confident (ECE 0.33) while preserving ranking (ROC 0.755);',
    'On unseen epitopes the network is poorly calibrated while preserving ranking '
    f'(expected calibration error [ECE] 0.254 {PM} 0.054, Brier 0.283 {PM} 0.033 at '
    f'ROC-AUC 0.747 {PM} 0.010);',
    expected=1, label='B3 Results, calibration')

# B4 — melanoma antigen epitopes
d.replace(
    'Among melanoma antigens, STEP performed best on MART-1/Melan-A epitopes '
    '(ROC-AUC 0.983), followed by gp100 (0.719)',
    'Among melanoma antigens, STEP performed best on MART-1/Melan-A epitopes '
    f'(ROC-AUC 0.960 {PM} 0.040 for AAGIGILTV), followed by gp100 '
    f'(0.794 {PM} 0.108)',
    expected=1, label='B4 Results, melanoma antigens')
i = para('reached ROC-AUC 0.910 on 1,582 melanoma')
d.replace_in_paragraph(i, 'MART-1 epitopes (0.983)',
                       f'MART-1 epitopes (0.960 {PM} 0.040)',
                       label='B4 Discussion, MART-1') if 'MART-1 epitopes (0.983)' \
    in before[i] else None

# B6 — Score-CAM binding probabilities
d.replace('NLVPMVATV (PDB 3GSN; P(bind) = 1.000)',
          f'NLVPMVATV (PDB 3GSN; P(bind) = 0.998 {PM} 0.004)',
          expected=1, label='B6 Results, 3GSN')
d.replace('NLVPMVATV (PDB 3GSN; predicted binding probability 1.000)',
          f'NLVPMVATV (PDB 3GSN; predicted binding probability 0.998 {PM} 0.004)',
          expected=1, label='B6 Fig 5 caption, 3GSN')
d.replace('EAAGIGILTV (PDB 4QOK; P(bind) = 0.969)',
          f'EAAGIGILTV (PDB 4QOK; P(bind) = 0.929 {PM} 0.122)',
          expected=1, label='B6 Results, 4QOK')
d.replace('EAAGIGILTV (PDB 4QOK; predicted binding probability 0.969)',
          f'EAAGIGILTV (PDB 4QOK; predicted binding probability 0.929 {PM} 0.122)',
          expected=1, label='B6 Fig 5 caption, 4QOK')

# ---------------------------------------------- A3: Methods 2.1 to v2 counts
d.replace(
    'The epitope-level split allocated 600 epitopes to training, 100 to validation '
    'and 581 to test, accounting for all 1,281 epitopes in the pool. The split '
    'yielded: a training set of 131,974 samples; a validation set of 5,243 samples '
    '(100 epitopes, 320 positive pairs, 6.1% positive)',
    'The epitope-level split allocated 600 epitopes to training, 100 to validation '
    'and 576 to test, accounting for all 1,276 epitopes in the pool. The split '
    'yielded: a training set of 130,067 samples; a validation set of 5,227 samples '
    '(100 epitopes, 308 positive pairs, 5.9% positive)',
    expected=1, label='A3 Methods 2.1, split counts')
d.replace('an imbalanced test set of 29,948 samples (5.5% positive, 581 epitopes); '
          'a balanced test set of 3,314 samples (50% positive, 581 epitopes)',
          'an imbalanced test set of 29,703 samples (5.6% positive, 576 epitopes); '
          'a balanced test set of 3,314 samples (50% positive, 576 epitopes)',
          expected=1, label='A3 Methods 2.1, test sets')
d.replace('STEP was trained on 131,974 TCRβ CDR3–epitope pairs (97,666 '
          'positive, 34,308 negative) spanning 600 epitopes',
          'STEP was trained on 130,067 TCRβ CDR3–epitope pairs (95,571 '
          'positive, 34,496 negative) spanning 600 epitopes',
          expected=1, label='A3 Results opening, training counts')
d.replace('We trained STEP on 97,666 positive and 34,308 negative '
          'TCRβ–epitope pairs spanning 600 epitopes',
          'We trained STEP on 95,571 positive and 34,496 negative '
          'TCRβ–epitope pairs spanning 600 epitopes',
          expected=1, label='A3 Introduction, training counts')

# ---------------------------------------------- new Methods sentences
d.replace(
    'Per-baseline provenance, repository URLs, commit hashes, training commands, and '
    'reproduction status are reported in Table S10.',
    'Per-baseline provenance, repository URLs, commit hashes, training commands, and '
    'reproduction status are reported in Table S10. Only the unseen-epitope '
    'partition was reconstructed under the corrected pipeline: the Seen-Pair, '
    'Unseen-TCR and Unseen-Pair partitions have no counterpart in the repaired '
    'split and were not rebuilt, and neither were the imbalanced evaluation of '
    'Section 3.2 or the leave-one-database-out analysis, both of which refer to the '
    'original split throughout.',
    expected=1, label='A2 Methods 2.5, partitions not reconstructed')

# ---------------------------------------------- R1-7 marker
d.replace(
    'Thirteen of the 31 melanoma epitopes are present in the training epitope set '
    f'{EM} including both MART-1/Melan-A epitopes, which together supply 88% of '
    f'melanoma positives {EM} and those thirteen carry 1,087 of the 1,582 melanoma '
    'rows (68.7%) [[PENDING: R1-7 ' + EM + ' these are the splits reported here; on the '
    'v2 splits the figures are 12 epitopes and 1,050 rows (66.4%), so update if the '
    'cancer evaluation is regenerated on v2]].',
    'Twelve of the 31 melanoma epitopes are present in the training epitope set '
    f'{EM} including both MART-1/Melan-A epitopes, which together supply 88% of '
    f'melanoma positives {EM} and those twelve carry 1,050 of the 1,582 melanoma '
    'rows (66.4%).',
    expected=1, label='R1-7 resolved to the v2 figures')

d.save()
print('manuscript saved')

# ======================================================================== SI
s = Docx(SI)
t7 = list(s.tree.iter(WTBL))[7]
S7 = {
    ('Balanced', 'STEP (Full)'): (f'0.747 {PM} 0.010', f'0.747 {PM} 0.011',
                                  f'0.671 {PM} 0.006', f'0.662 {PM} 0.019',
                                  f'0.210 {PM} 0.198'),
    ('Balanced', 'No structure'): (f'0.754 {PM} 0.009', f'0.733 {PM} 0.010',
                                   f'0.681 {PM} 0.008', f'0.708 {PM} 0.020',
                                   f'0.027 {PM} 0.010'),
    ('Melanoma', 'STEP (Full)'): (f'0.913 {PM} 0.011', f'0.894 {PM} 0.030',
                                  f'0.752 {PM} 0.037', f'0.795 {PM} 0.023',
                                  f'0.210 {PM} 0.198'),
    ('Melanoma', 'No structure'): (f'0.894 {PM} 0.006', f'0.850 {PM} 0.002',
                                   f'0.627 {PM} 0.031', f'0.724 {PM} 0.016',
                                   f'0.027 {PM} 0.010'),
}
done = set()
for tr in list(t7.iter(WTR))[1:]:
    cs = cells(tr)
    key = tuple(' '.join(''.join(x.text or '' for x in c.iter(f'{{{W}}}t')).split())
                for c in cs[:2])
    if key in S7:
        for k, v in enumerate(S7[key], start=2):
            set_cell(cs[k], v)
        done.add(key)
    elif key[1] == '300 epitopes':
        set_cell(cs[1], '300 epitopes (original pipeline)')
    else:
        raise AssertionError(f'unexpected Table S7 row {key}')
assert done == set(S7), f'missed {set(S7) - done}'
s.save()
print('supplementary saved')

# ---------------------------------------------------------------- verify
d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
import difflib
for k, (b, a) in enumerate(zip(before, after)):
    if b != a:
        print(f'\n===== [{k}] =====')
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, b, a, autojunk=False).get_opcodes():
            if tag == 'equal':
                continue
            if b[i1:i2]:
                print(f'  - {b[i1:i2][:400]}')
            if a[j1:j2]:
                print(f'  + {a[j1:j2][:400]}')
full = '\n'.join(after)
for bad in ('PENDING: R1-7', 'Seen-Pair (regular cross-validation)',
            '0.739–0.771', 'ECE 0.33'):
    assert bad not in full, f'still present: {bad!r}'
import re
print('\nmarkers left:', re.findall(r'\[\[[^\]]+\]\]', full))
print('abstract words:', len(after[10].split()))
import zipfile
for f in (MS, SI):
    with zipfile.ZipFile(f) as z:
        assert z.testzip() is None
print('zips OK; paragraphs', len(after))
