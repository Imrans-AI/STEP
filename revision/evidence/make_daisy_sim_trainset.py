"""
Rebuild the DAISY training file for the similarity-disjoint split (R2-1 Part C).

baselines/repos/DAISY is a nested upstream clone and is not tracked by this
repository, so the CSV itself cannot be committed; this script regenerates it.

DAISY's train.py takes POSITIVE pairs only and samples its own negatives from
small_healthy_tcr.csv, so the file is the positive rows of the sim training split
with the column names DAISY expects (peptide, cdr3), in split order.

Run:  python revision/evidence/make_daisy_sim_trainset.py [--check]
"""
import sys
from pathlib import Path

import pandas as pd

SRC = Path('data/splits_600_epitope_sim/train.tsv')
DST = Path('baselines/repos/DAISY/data_ours/tcr_peptide_train_sim.csv')

d = pd.read_csv(SRC, sep='\t')
pos = d[d.label == 1]
out = pd.DataFrame({'peptide': pos['antigen.epitope'].values, 'cdr3': pos['cdr3'].values})
print(f"{SRC}: {len(d):,} rows -> {len(out):,} positives, "
      f"{out.peptide.nunique()} epitopes, {out.cdr3.nunique():,} CDR3b")

if '--check' in sys.argv:
    have = pd.read_csv(DST)
    same = have.equals(out)
    print(f"{DST} exists: {DST.exists()}  identical to regenerated: {same}")
    sys.exit(0 if same else 1)

DST.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(DST, index=False)
print(f"wrote {DST}")
