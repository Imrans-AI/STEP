"""
R2-C4 / R2-C5 / R2-C6: three read-only figure-dependent lookups.

Nothing is written outside revision/evidence/. No model is loaded and no file
under data/, models/ or results/ is modified.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = '/home/muhammad-imran/PhD_data/TCR_Module_1'
os.chdir(ROOT)
OUT = 'revision/evidence/r2_c4_c5_c6_lookups.json'
rep = {}
sep = "=" * 78

AA3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
       'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
       'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
       'Y': 'TYR', 'V': 'VAL'}

# ===================================================== R2-C4
print(sep); print("  R2-C4  SCORE-CAM TOP-ATTENTION POSITIONS"); print(sep)
CAM = 'results/fig6_scorecam'
meta = pd.read_csv(f'{CAM}/metadata.csv').set_index('epitope')
rep['c4'] = {}


def analyse(tag, cam_file, cdr3, epi, cdr3_start, epi_start, note):
    cam = np.load(f'{CAM}/{cam_file}')
    nc, ne = len(cdr3), len(epi)
    t = cam[:nc, :ne]
    print(f"\n--- {tag} ---")
    print(f"  file {cam_file}  raw shape {cam.shape} -> trimmed {t.shape} "
          f"(CDR3b {nc} aa, epitope {ne} aa)")
    print(f"  {note}")
    print(f"  CAM range {t.min():.4f}-{t.max():.4f}, mean {t.mean():.4f}")

    ecol = t.sum(axis=0)
    emax = t.max(axis=0)
    rrow = t.sum(axis=1)
    rmax = t.max(axis=1)

    print(f"\n  EPITOPE positions (column attention), numbering from {epi_start}:")
    print(f"    {'pos':>5} {'res':>8} {'sum':>9} {'max':>8} {'share':>7}")
    eo = np.argsort(-ecol)
    for j in range(ne):
        mark = '  <-- top' if j in eo[:3] else ''
        print(f"    {epi_start + j:>5} {AA3[epi[j]] + '-' + str(epi_start + j):>8} "
              f"{ecol[j]:>9.4f} {emax[j]:>8.4f} {100 * ecol[j] / ecol.sum():>6.1f}%{mark}")
    etop = [f"{AA3[epi[j]]}-{epi_start + j}" for j in eo[:3]]
    print(f"    TOP 3 epitope positions: {', '.join(etop)}")

    print(f"\n  CDR3b positions (row attention), numbering from {cdr3_start}:")
    print(f"    {'pos':>5} {'res':>8} {'sum':>9} {'max':>8} {'share':>7}")
    ro = np.argsort(-rrow)
    for i in range(nc):
        mark = '  <-- top' if i in ro[:3] else ''
        print(f"    {cdr3_start + i:>5} {AA3[cdr3[i]] + '-' + str(cdr3_start + i):>8} "
              f"{rrow[i]:>9.4f} {rmax[i]:>8.4f} {100 * rrow[i] / rrow.sum():>6.1f}%{mark}")
    rtop = [f"{AA3[cdr3[i]]}-{cdr3_start + i}" for i in ro[:3]]
    print(f"    TOP 3 CDR3b positions: {', '.join(rtop)}")

    i, j = np.unravel_index(np.argmax(t), t.shape)
    print(f"    single hottest cell: CDR3b {AA3[cdr3[i]]}-{cdr3_start + i} x "
          f"epitope {AA3[epi[j]]}-{epi_start + j}  = {t[i, j]:.4f}")
    rep['c4'][tag] = dict(
        cam_file=cam_file, cdr3=cdr3, epitope=epi, trimmed_shape=list(t.shape),
        epitope_top3=etop, cdr3_top3=rtop,
        epitope_order=[f"{AA3[epi[j]]}-{epi_start + j}" for j in eo],
        cdr3_order=[f"{AA3[cdr3[i]]}-{cdr3_start + i}" for i in ro],
        hottest_cell=dict(cdr3=f"{AA3[cdr3[i]]}-{cdr3_start + i}",
                          epitope=f"{AA3[epi[j]]}-{epi_start + j}", value=float(t[i, j])),
        note=note)
    return t


# Panel a/b: 3GSN / NLVPMVATV. CAM's own CDR3b (metadata) is CASSKTQLWETQYF.
analyse('3GSN_NLVPMVATV_cam_own_cdr3',
        'NLVPMVATV_cam.npy', 'CASSKTQLWETQYF', 'NLVPMVATV', 95, 1,
        "CDR3b from metadata.csv (RA14, CASSKTQLWETQYF); numbering 95.. is the "
        "figure's axis, which FIG6_NATURE_V4 takes from 3GSN, not from this CDR3b.")

# The figure labels those 14 rows with 3GSN chain B residues 95-108.
analyse('3GSN_NLVPMVATV_as_labelled_on_figure',
        'NLVPMVATV_cam.npy', 'PVTGGIYGYTFGSG', 'NLVPMVATV', 95, 1,
        "Same CAM, rows relabelled with 3GSN chain B 95-108 "
        "(PRO95..GLY108) exactly as the submitted figure prints them.")

# Panel c/d as submitted: AAGIGILTV_cam.npy (5HHO/DMF5) drawn with 4QOK labels.
analyse('fig5_panel_c_as_submitted',
        'AAGIGILTV_cam.npy', 'ETGLGTGELFFG', 'AAGIGILTV', 95, 2,
        "AS SUBMITTED: CAM is AAGIGILTV (metadata: 5HHO / DMF5 / ASSHAGLGVEQY, "
        "conf 0.969) but rows are labelled with 4QOK residues 95-106 and columns "
        "with 4QOK epitope positions 2-10. Values and labels are from different TCRs.")

# The genuine 4QOK CAM, written for Fig 6 and never used in Fig 5.
t4 = analyse('4QOK_EAAGIGILTV_genuine',
             'EAAGIGILTV_4QOK_cam.npy', 'CAWSETGLGTGELFF', 'EAAGIGILTV', 1, 1,
             "The real 4QOK/EAAGIGILTV CAM (written 2026-07-03 for Fig 6). "
             "CDR3b CAWSETGLGTGELFF, 15 aa; epitope EAAGIGILTV, 10 aa. "
             "Sequence numbering 1.. ; Fig 6 panel a prints the CDR3b as "
             "single letters C,A,W,S,E,T,G,L,G,T,G,E,L,F,F.")

# true contacts, for the same pair
d = np.load(f'{CAM}/EAAGIGILTV_4QOK_actual_dist.npy')[:15, :10]
print("\n--- 4QOK true Ca-Ca distances (same pair) ---")
cdr3, epi = 'CAWSETGLGTGELFF', 'EAAGIGILTV'
cc = (d < 8.0)
print(f"  cells < 8 A: {int(cc.sum())} of {d.size}")
print(f"  epitope positions by number of CDR3b contacts < 8 A:")
for j in np.argsort(-cc.sum(axis=0)):
    print(f"    {AA3[epi[j]]}-{j + 1:<3} {int(cc[:, j].sum()):>3} contacts   "
          f"min dist {d[:, j].min():.2f} A")
print(f"  CDR3b positions by number of epitope contacts < 8 A:")
for i in np.argsort(-cc.sum(axis=1)):
    if cc[i].sum():
        print(f"    {AA3[cdr3[i]]}-{i + 1:<3} {int(cc[i].sum()):>3} contacts   "
              f"min dist {d[i].min():.2f} A")
rep['c4']['4QOK_true_contacts'] = dict(
    n_cells_under_8A=int(cc.sum()),
    epitope_contact_counts={f"{AA3[epi[j]]}-{j + 1}": int(cc[:, j].sum()) for j in range(10)},
    cdr3_contact_counts={f"{AA3[cdr3[i]]}-{i + 1}": int(cc[i].sum()) for i in range(15)})

# ===================================================== R2-C5
print("\n" + sep); print("  R2-C5  DAISY PR-AUC IN FIG 3"); print(sep)
src = 'results/ALL_MODELS_PER_SAMPLE.csv'
print(f"  Fig 3 script: FIG_MERGED_3x3.py; panel c (PR curves) and panel d (AUPR bars)")
p = pd.read_csv(src)
print(f"  panel-c input {src}: {len(p):,} rows, columns {list(p.columns)}")
rep['c5'] = {'per_sample_file': src, 'n': int(len(p))}
lab = 'label' if 'label' in p.columns else p.columns[2]
y = p[lab].values.astype(int)
for c in p.columns:
    if c.lower() in ('cdr3', 'antigen.epitope', 'epitope', lab.lower()):
        continue
    try:
        v = p[c].values.astype(float)
    except (ValueError, TypeError):
        continue
    print(f"    {c:<22} ROC {roc_auc_score(y, v):.4f}   PR (average_precision) "
          f"{average_precision_score(y, v):.4f}")
    rep['c5'].setdefault('recomputed', {})[c] = dict(
        roc=float(roc_auc_score(y, v)), pr=float(average_precision_score(y, v)))

for f in ['results/ALL_6_MODELS_BALANCED_FINAL.csv',
          'results/baseline_rerun_v2/summary_v2.csv']:
    if os.path.exists(f):
        print(f"\n  {f}:")
        print(pd.read_csv(f).to_string(index=False))

# ===================================================== R2-C6
print("\n" + sep); print("  R2-C6  LENGTH COMPOSITION OF THE PEPTIDE SCAN"); print(sep)
s = pd.read_csv('results/mel5_scan_scores.csv')
print(f"  results/mel5_scan_scores.csv: {len(s):,} rows, columns {list(s.columns)}")
col = 'peptide' if 'peptide' in s.columns else s.columns[0]
s['length'] = s[col].astype(str).str.len()
vc = s['length'].value_counts().sort_index()
print(f"\n  {'length':>7} {'n':>7} {'share':>8}")
for L, n in vc.items():
    print(f"  {L:>7} {n:>7,} {100 * n / len(s):>7.2f}%")
print(f"  TOTAL {len(s):,}")
dups = s[col].duplicated().sum()
print(f"  distinct peptides {s[col].nunique():,} (duplicate rows {dups})")
rep['c6'] = dict(n_rows=int(len(s)), n_distinct=int(s[col].nunique()),
                 by_length={int(L): int(n) for L, n in vc.items()})
print("\n  named peptides:")
for q in ['EAAGIGILTV', 'LLLGIGILVL', 'AAGIGILTV', 'NLSALGIFST', 'ELAGIGILTV']:
    r = s[s[col] == q]
    if len(r):
        row = r.iloc[0]
        extra = ' '.join(f"{c}={row[c]}" for c in s.columns
                         if c not in (col, 'length'))
        print(f"    {q:<12} length {len(q):>2}  present  {extra}")
    else:
        print(f"    {q:<12} length {len(q):>2}  ABSENT from the file")
    rep['c6'].setdefault('named', {})[q] = dict(length=len(q), present=bool(len(r)))

json.dump(rep, open(OUT, 'w'), indent=2, default=str)
print(f"\nsaved {OUT}")
