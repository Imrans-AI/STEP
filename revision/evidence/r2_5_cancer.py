"""
R2-5 (inference only, no training): verify the reviewer's four claims about the
cancer results and produce the per-epitope analysis they ask for.

Scores data/cancer_specific/{melanoma,lung}_test.tsv with the CORRECTED STEP model
(models/checkpoints_v2/corrected_seed{42,43,44}, prior structure_priors_v2.npz),
strict=True loading asserted. Physicochemical maps come from the v1 index, which
covers every cancer pair; a sample is re-derived from scratch and compared, so the
lookup is checked rather than trusted.

Nothing is trained and nothing outside revision/evidence + results/r2_5_cancer is
written.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import Levenshtein
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'models'))
os.chdir(ROOT)

from step_model import STEPModel                      # noqa: E402
from dataset_physchem_only import PhysChemOnlyDataset  # noqa: E402

V2TRAIN = 'data/splits_600_epitope_v2/train.tsv'
SETS = {'melanoma': 'data/cancer_specific/melanoma_test.tsv',
        'lung': 'data/cancer_specific/lung_test.tsv'}
SEEDS = (42, 43, 44)
OUTDIR = 'results/r2_5_cancer'
OUTJ = 'revision/evidence/r2_5_cancer.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIN_N, MHC1_MAX_LEN, EPI_WINDOW = 5, 15, 15
sep = "=" * 78
rep = {}

np.random.seed(42)
torch.manual_seed(42)

train = pd.read_csv(V2TRAIN, sep='\t')
train_cdr3 = set(train.cdr3)
train_cdr3_pos = set(train.loc[train.label == 1, 'cdr3'])
train_epi = sorted(set(train['antigen.epitope']))
train_pairs = set(zip(train.cdr3, train['antigen.epitope']))
print(sep)
print(f"v2 training set: {len(train):,} rows, {len(train_epi)} epitopes, "
      f"{len(train_cdr3):,} distinct CDR3b "
      f"({len(train_cdr3_pos):,} occurring as positives)")

# =============================================================== score STEP
print("\n" + sep)
print("  SCORING WITH THE CORRECTED STEP MODEL (inference only)")
print(sep)
PRED = {}
for name, path in SETS.items():
    ds = PhysChemOnlyDataset(path)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    cols = []
    for s in SEEDS:
        ck = torch.load(f'models/checkpoints_v2/corrected_seed{s}/best_model.pt',
                        map_location='cpu', weights_only=False)
        m = STEPModel(use_structure_priors=True, priors_file='structure_priors_v2.npz')
        msd, sd_ = m.state_dict(), ck['model_state_dict']
        assert set(msd) == set(sd_), f"key mismatch seed {s}"
        assert all(msd[k].shape == sd_[k].shape for k in msd), f"shape mismatch seed {s}"
        m.load_state_dict(sd_)                        # strict=True
        m.to(DEVICE).eval()
        out, yl = [], []
        with torch.no_grad():
            for pc, gf, y in dl:
                out.extend(m(pc.to(DEVICE), gf.to(DEVICE)).cpu().numpy().ravel())
                yl.extend(y.numpy().ravel())
        del m
        cols.append(np.asarray(out, float))
        assert np.array_equal(np.asarray(yl, int), ds.data.label.values.astype(int))
    d = pd.read_csv(path, sep='\t')
    d['STEP_seed42'], d['STEP_seed43'], d['STEP_seed44'] = cols
    d['STEP_ensemble'] = np.mean(cols, axis=0)
    PRED[name] = d
    print(f"  {name:<9} n={len(d):>5}  {len(SEEDS)} seeds x 163/163 keys, strict load OK")

# Verify the physchem lookup rather than trusting it. data/compute_physchem.py is a
# SCRIPT: importing it runs its main body, which writes embeddings/physchem_maps.npy
# relative to the current directory. Only the property tables and the map function are
# wanted, so the file is exec'd up to the end of that function and no further.
_src = open('data/compute_physchem.py').read().split(
    '\n# Collect unique CDR3β-epitope pairs from all splits')[0]
assert 'def compute_physchem_map' in _src and 'np.save' not in _src
_ns = {}
exec(compile(_src, 'data/compute_physchem.py', 'exec'), _ns)
compute_physchem_map = _ns['compute_physchem_map']
chk = PRED['melanoma'].sample(n=25, random_state=42)
dsm = PhysChemOnlyDataset(SETS['melanoma'])
bad = 0
for i in chk.index:
    a = dsm[i][0].numpy()
    b = np.asarray(compute_physchem_map(chk.loc[i, 'cdr3'],
                                        chk.loc[i, 'antigen.epitope']), dtype=np.float32)
    if not np.allclose(a, b, atol=1e-5):
        bad += 1
print(f"  physchem lookup re-derived from scratch for 25 random melanoma pairs: "
      f"{25 - bad}/25 identical")
rep['physchem_lookup_check'] = dict(n=25, identical=25 - bad)

# ================================================== 1. CDR3b training overlap
print("\n" + sep)
print("  1. WHAT FRACTION OF CANCER CDR3b ARE IN THE v2 TRAINING SET?")
print(sep)
ov = {}
print(f"  {'set':<9} {'subset':<10} {'rows':>6} {'rows w/ seen CDR3b':>19} "
      f"{'distinct CDR3b':>15} {'distinct seen':>14}")
for name, d in PRED.items():
    ov[name] = {}
    for sub, m in (('all', d.label >= 0), ('positives', d.label == 1),
                   ('negatives', d.label == 0)):
        x = d[m]
        seen_rows = x.cdr3.isin(train_cdr3)
        u = set(x.cdr3)
        us = u & train_cdr3
        ov[name][sub] = dict(rows=int(len(x)), rows_seen=int(seen_rows.sum()),
                             rows_seen_frac=float(seen_rows.mean()),
                             distinct=len(u), distinct_seen=len(us),
                             distinct_seen_frac=len(us) / len(u))
        print(f"  {name:<9} {sub:<10} {len(x):>6} "
              f"{seen_rows.sum():>8} ({seen_rows.mean():>6.1%}) "
              f"{len(u):>15} {len(us):>7} ({len(us) / len(u):>6.1%})")
rep['cdr3_overlap'] = ov

# ================================================== 2. positives per epitope
print("\n" + sep)
print("  2. POSITIVES PER EPITOPE  —  the \"two MART-1 epitopes supply 88%\" claim")
print(sep)
pe = {}
for name, d in PRED.items():
    p = d[d.label == 1]['antigen.epitope'].value_counts()
    tot = int(p.sum())
    top2 = p.head(2)
    pe[name] = dict(total_positives=tot,
                    per_epitope={k: int(v) for k, v in p.items()},
                    top2=list(top2.index), top2_n=int(top2.sum()),
                    top2_frac=float(top2.sum() / tot))
    print(f"  {name}: {tot} positives over {len(p)} epitopes; "
          f"top two = {', '.join(top2.index)} "
          f"= {int(top2.sum())}/{tot} = {top2.sum() / tot:.1%}")
rep['positives_per_epitope'] = pe

# ================================================== 3/4. per-epitope + distance
print("\n" + sep)
print(f"  3+4. PER-EPITOPE ROC-AUC (>= {MIN_N} samples, both classes) AND DISTANCE")
print(sep)
MODELS = ['STEP_seed42', 'STEP_ensemble']
tabs = {}
for name, d in PRED.items():
    rows = []
    for e, g in d.groupby('antigen.epitope'):
        npos, n = int(g.label.sum()), len(g)
        dist = min(Levenshtein.distance(e, t) for t in train_epi)
        nearest = min(train_epi, key=lambda t: (Levenshtein.distance(e, t), t))
        r = dict(epitope=e, length=len(e), n=n, n_pos=npos, n_neg=n - npos,
                 evaluable=bool(n >= MIN_N and 0 < npos < n),
                 min_dist=dist, exact_in_train=bool(e in set(train_epi)),
                 nearest_train=nearest,
                 identity=round(1 - dist / max(len(e), len(nearest)), 3),
                 cdr3_seen_frac=float(g.cdr3.isin(train_cdr3).mean()))
        for mo in MODELS:
            r[mo] = (float(roc_auc_score(g.label, g[mo]))
                     if r['evaluable'] else None)
        rows.append(r)
    T = pd.DataFrame(rows).sort_values('n', ascending=False)
    tabs[name] = T
    pooled = {mo: float(roc_auc_score(d.label, d[mo])) for mo in MODELS}
    pooled_pr = {mo: float(average_precision_score(d.label, d[mo])) for mo in MODELS}
    ev = T[T.evaluable]
    print(f"\n  --- {name.upper()} ---")
    print(f"  pooled   ROC {pooled['STEP_seed42']:.4f} (seed 42) / "
          f"{pooled['STEP_ensemble']:.4f} (ensemble)   "
          f"PR {pooled_pr['STEP_seed42']:.4f} / {pooled_pr['STEP_ensemble']:.4f}")
    print(f"  evaluable epitopes: {len(ev)} of {len(T)}  "
          f"(carrying {int(ev.n.sum())} of {len(d)} rows)")
    if len(ev):
        print(f"  per-epitope mean ROC: {ev['STEP_seed42'].mean():.4f} (seed 42) / "
              f"{ev['STEP_ensemble'].mean():.4f} (ensemble)   "
              f"median {ev['STEP_seed42'].median():.4f} / "
              f"{ev['STEP_ensemble'].median():.4f}")
        print(f"  GAP pooled - per-epitope mean: "
              f"{pooled['STEP_seed42'] - ev['STEP_seed42'].mean():+.4f} (seed 42)")
    print(f"\n  {'epitope':<26}{'len':>4}{'n':>5}{'pos':>5}{'dist':>5}{'exact':>6}"
          f"{'CDR3b seen':>11}{'seed42':>9}{'ensemble':>10}")
    for _, r in T.iterrows():
        f42 = f"{r['STEP_seed42']:.3f}" if pd.notna(r['STEP_seed42']) else "n/a"
        fen = f"{r['STEP_ensemble']:.3f}" if pd.notna(r['STEP_ensemble']) else "n/a"
        print(f"  {r.epitope:<26}{r.length:>4}{r.n:>5}{r.n_pos:>5}{r.min_dist:>5}"
              f"{'YES' if r.exact_in_train else '-':>6}{r.cdr3_seen_frac:>10.0%}"
              f"{f42:>9}{fen:>10}")
    T.to_csv(f'{OUTDIR}/per_epitope_{name}.csv', index=False) if os.path.isdir(OUTDIR) \
        else None
    rep.setdefault('per_epitope', {})[name] = dict(
        pooled_roc=pooled, pooled_pr=pooled_pr,
        n_evaluable=int(len(ev)), n_epitopes=int(len(T)),
        rows_in_evaluable=int(ev.n.sum()),
        mean_roc={mo: (float(ev[mo].mean()) if len(ev) else None) for mo in MODELS},
        median_roc={mo: (float(ev[mo].median()) if len(ev) else None) for mo in MODELS},
        table=T.where(pd.notnull(T), None).to_dict('records'))

    # how much of the pooled figure is the top-two-epitope block carrying?
    top2 = list(d[d.label == 1]['antigen.epitope'].value_counts().head(2).index)
    rest = d[~d['antigen.epitope'].isin(top2)]
    if rest.label.nunique() == 2:
        drop = {mo: float(roc_auc_score(rest.label, rest[mo])) for mo in MODELS}
        print(f"  pooled with the two largest epitopes ({', '.join(top2)}) REMOVED: "
              f"n={len(rest)} ({int(rest.label.sum())} pos) "
              f"ROC {drop['STEP_seed42']:.4f} / {drop['STEP_ensemble']:.4f}")
        rep['per_epitope'][name]['pooled_without_top2'] = dict(
            dropped=top2, n=int(len(rest)), pos=int(rest.label.sum()), roc=drop)

# ================================================== 5. Fig S8 contents
print("\n" + sep)
print("  5. FIGURE S8 CONTENTS")
print(sep)
s8 = {}
for name, d in PRED.items():
    vc = d['antigen.epitope'].value_counts()
    longs = [(e, len(e), int(vc[e])) for e in vc.index if len(e) > MHC1_MAX_LEN]
    s8[name] = dict(
        n_epitopes=int(len(vc)),
        over_15mer=[dict(epitope=e, length=L, n=n) for e, L, n in longs],
        rows_over_15mer=int(sum(n for _, _, n in longs)),
        has_GILGFVFTL=bool('GILGFVFTL' in vc.index),
        GILGFVFTL_n=int(vc.get('GILGFVFTL', 0)),
        GILGFVFTL_pos=int(d[(d['antigen.epitope'] == 'GILGFVFTL') & (d.label == 1)].shape[0]))
    print(f"  {name}: {len(vc)} epitopes; longer than {MHC1_MAX_LEN} aa: "
          f"{len(longs)} epitopes / {sum(n for _, _, n in longs)} rows")
    for e, L, n in longs:
        print(f"      {e:<28} length {L:>2}  n {n:>3}   "
              f"(epitope window is {EPI_WINDOW}; last {L - EPI_WINDOW} residues truncated)")
    print(f"      GILGFVFTL present: {s8[name]['has_GILGFVFTL']}"
          + (f"  n={s8[name]['GILGFVFTL_n']} ({s8[name]['GILGFVFTL_pos']} pos)"
             if s8[name]['has_GILGFVFTL'] else ""))
rep['figS8'] = s8

# ================================================== 6. overlap-free, v2
print("\n" + sep)
print("  6. OVERLAP-FREE SUBSETS RECOMPUTED AGAINST v2 TRAINING")
print(sep)
of = {}
for name, d in PRED.items():
    of[name] = {}
    defs = {
        'as_submitted_pair_level': ~pd.Series(
            [(c, e) in train_pairs for c, e in zip(d.cdr3, d['antigen.epitope'])],
            index=d.index),
        'cdr3_level': ~d.cdr3.isin(train_cdr3),
        'cdr3_and_epitope_level': (~d.cdr3.isin(train_cdr3))
        & (~d['antigen.epitope'].isin(set(train_epi))),
    }
    print(f"\n  --- {name.upper()} (full set: n={len(d)}, "
          f"{int(d.label.sum())} pos) ---")
    for k, m in defs.items():
        x = d[m]
        npos, nneg = int(x.label.sum()), int((x.label == 0).sum())
        rec = dict(n=int(len(x)), pos=npos, neg=nneg)
        if npos and nneg:
            for mo in MODELS:
                rec[mo] = float(roc_auc_score(x.label, x[mo]))
            # balanced version, matching how the paper reports it
            pos = x[x.label == 1]
            neg = x[x.label == 0].sample(n=min(npos, nneg), random_state=42)
            b = pd.concat([pos, neg])
            rec['balanced_n'] = int(len(b))
            for mo in MODELS:
                rec[f'balanced_{mo}'] = float(roc_auc_score(b.label, b[mo]))
            print(f"  {k:<26} n={len(x):>5} ({npos} pos / {nneg} neg)  "
                  f"ROC {rec['STEP_seed42']:.4f} / {rec['STEP_ensemble']:.4f}  | "
                  f"balanced n={rec['balanced_n']:>4} "
                  f"ROC {rec['balanced_STEP_seed42']:.4f} / "
                  f"{rec['balanced_STEP_ensemble']:.4f}")
        else:
            print(f"  {k:<26} n={len(x):>5} ({npos} pos / {nneg} neg)  "
                  f"-- not evaluable, one class empty")
        of[name][k] = rec
rep['overlap_free_v2'] = of

os.makedirs(OUTDIR, exist_ok=True)
for name, T in tabs.items():
    T.to_csv(f'{OUTDIR}/per_epitope_{name}.csv', index=False)
    PRED[name].to_csv(f'{OUTDIR}/predictions_{name}.csv', index=False)
json.dump(rep, open(OUTJ, 'w'), indent=2, default=str)
print(f"\nsaved {OUTJ}\nsaved {OUTDIR}/")
