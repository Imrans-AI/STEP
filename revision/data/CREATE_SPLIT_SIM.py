"""
CREATE_SPLIT_SIM.py — similarity-disjoint rebuild of the 600-epitope split (reviewer R2-1).

R2-1: "Exact-string disjointness is not 'unseen'. Re-split by sequence similarity."

Same pool, same design and same seed as CREATE_SPLIT_v2.py (600 train epitopes,
100 validation, remainder test, plus a balanced test subset). The one change is
that epitopes are first grouped into similarity CLUSTERS, and a cluster is
assigned to exactly one split, so no near-duplicate can straddle a boundary.

Linkage rule (either is sufficient, applied transitively / single-linkage):
    Levenshtein distance <= 2        OR        identity >= 80%
where identity = 1 - distance / max(len_a, len_b).

The pool is reconstructed as train+val+test of splits_600_epitope_v2, which is
exactly the cleaned pool that script produced (its split() partitions every row
by epitope, so the three parts reassemble the whole).

Writes data/splits_600_epitope_sim/. Nothing else is opened for writing.

Run:  python data/CREATE_SPLIT_SIM.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import Levenshtein

HERE = Path(__file__).parent
V2 = HERE / 'splits_600_epitope_v2'
OUT = HERE / 'splits_600_epitope_sim'

SEED = 42
N_TRAIN_EPI = 600
N_VAL_EPI = 100
MAX_DIST = 2
MIN_IDENT = 0.80
sep = "=" * 78


def linked(a, b):
    d = Levenshtein.distance(a, b)
    return d <= MAX_DIST or (1.0 - d / max(len(a), len(b))) >= MIN_IDENT


# ------------------------------------------------------------------ 1. pool
print(sep); print("  1. POOL (= train + val + test of splits_600_epitope_v2)"); print(sep)
parts = {k: pd.read_csv(V2 / f'{k}.tsv', sep='\t') for k in ['train', 'val', 'test']}
pool = pd.concat(parts.values(), ignore_index=True)
for k, d in parts.items():
    print(f"  v2 {k:6s} {len(d):>7,} rows  {d['antigen.epitope'].nunique():>5} epitopes")
print(f"  pooled {len(pool):>9,} rows  {pool['antigen.epitope'].nunique():>5} epitopes  "
      f"{pool['cdr3'].nunique():,} CDR3b  {pool.label.mean():.1%} positive")
epitopes = sorted(pool['antigen.epitope'].unique())
n = len(epitopes)

# ------------------------------------------------------------------ 2. cluster
print("\n" + sep)
print(f"  2. SIMILARITY CLUSTERING  (distance <= {MAX_DIST} OR identity >= {MIN_IDENT:.0%})")
print(sep)
parent = list(range(n))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def union(i, j):
    a, b = find(i), find(j)
    if a != b:
        parent[b] = a


# bucket by length: an edge needs |len difference| <= MAX_DIST, and the identity
# rule cannot fire beyond that either (identity >= 0.8 implies d <= 0.2*maxlen <= 3)
by_len = {}
for i, e in enumerate(epitopes):
    by_len.setdefault(len(e), []).append(i)
lengths = sorted(by_len)
n_edges = 0
for li in lengths:
    for lj in lengths:
        if lj < li or lj - li > 3:
            continue
        same = (li == lj)
        for i in by_len[li]:
            for j in by_len[lj]:
                # within one length bucket, visit each unordered pair once; across two
                # different buckets every (i, j) is already a distinct unordered pair,
                # so an index guard there would silently drop half of them
                if same and j <= i:
                    continue
                if linked(epitopes[i], epitopes[j]):
                    union(i, j)
                    n_edges += 1
print(f"  epitopes {n} | linking edges found {n_edges:,}")

comp = {}
for i in range(n):
    comp.setdefault(find(i), []).append(i)
clusters = [[epitopes[i] for i in v] for v in comp.values()]
clusters.sort(key=len, reverse=True)
sizes = pd.Series([len(c) for c in clusters])
print(f"  clusters: {len(clusters)}")
print(f"  size distribution: 1 -> {int((sizes == 1).sum())}, 2 -> {int((sizes == 2).sum())}, "
      f"3-5 -> {int(sizes.between(3, 5).sum())}, 6-10 -> {int(sizes.between(6, 10).sum())}, "
      f">10 -> {int((sizes > 10).sum())}")
print(f"  largest cluster {int(sizes.max())} epitopes; "
      f"median {int(sizes.median())}; mean {sizes.mean():.2f}")
print(f"  singletons carry {int((sizes == 1).sum())}/{n} epitopes "
      f"({100 * (sizes == 1).sum() / n:.1f}%)")
print("\n  five largest clusters:")
for c in clusters[:5]:
    ex = ', '.join(sorted(c)[:6])
    print(f"    {len(c):4d} epitopes:  {ex}{' …' if len(c) > 6 else ''}")

epi2cl = {e: ci for ci, c in enumerate(clusters) for e in c}
pool['cluster'] = pool['antigen.epitope'].map(epi2cl)

# ------------------------------------------------------------------ 3. split
print("\n" + sep)
print("  3. SPLIT BY CLUSTER (same design and seed as CREATE_SPLIT_v2.py)")
print(sep)
cl_rows = pool.groupby('cluster').size()
cl_epi = pool.groupby('cluster')['antigen.epitope'].nunique()

# train: largest clusters by row count first, mirroring "600 most frequent epitopes",
# until the TRAIN EPITOPE BUDGET is reached. A cluster is never split.
order = cl_rows.sort_values(ascending=False).index.tolist()
train_cl, got = [], 0
for ci in order:
    if got >= N_TRAIN_EPI:
        break
    train_cl.append(ci)
    got += int(cl_epi[ci])
rest = [ci for ci in order if ci not in set(train_cl)]
rng = np.random.RandomState(SEED)
rng.shuffle(rest)
val_cl, got_v = [], 0
for ci in rest:
    if got_v >= N_VAL_EPI:
        break
    val_cl.append(ci)
    got_v += int(cl_epi[ci])
test_cl = [ci for ci in rest if ci not in set(val_cl)]
assign = {'train': set(train_cl), 'val': set(val_cl), 'test': set(test_cl)}
print(f"  clusters -> train {len(train_cl)} | val {len(val_cl)} | test {len(test_cl)}"
      f"  (total {len(train_cl) + len(val_cl) + len(test_cl)} = {len(clusters)})")
print(f"  epitope budget: train target {N_TRAIN_EPI} got {got}; "
      f"val target {N_VAL_EPI} got {got_v}  (whole clusters, so exact targets are not reachable)")

out = {k: pool[pool.cluster.isin(v)].drop(columns='cluster').reset_index(drop=True)
       for k, v in assign.items()}

pos = out['test'][out['test'].label == 1]
neg = out['test'][out['test'].label == 0]
nb = neg.sample(n=min(len(pos), len(neg)), random_state=SEED)
out['test_balanced'] = (pd.concat([pos, nb], ignore_index=True)
                        .sample(frac=1, random_state=SEED).reset_index(drop=True))

print("\n" + sep); print("  4. PER-SPLIT SUMMARY"); print(sep)
print(f"  {'split':16s} {'rows':>9s} {'epitopes':>9s} {'clusters':>9s} {'CDR3b':>9s} "
      f"{'pos':>8s} {'pos frac':>9s}")
for k, d in out.items():
    ncl = d['antigen.epitope'].map(epi2cl).nunique()
    print(f"  {k:16s} {len(d):>9,} {d['antigen.epitope'].nunique():>9,} {ncl:>9,} "
          f"{d['cdr3'].nunique():>9,} {int(d.label.sum()):>8,} {d.label.mean():>8.1%}")

# ------------------------------------------------------------------ 5. verify
print("\n" + sep); print("  5. VERIFICATION"); print(sep)
E = {k: set(out[k]['antigen.epitope']) for k in ['train', 'val', 'test']}
C = {k: set(out[k]['antigen.epitope'].map(epi2cl)) for k in ['train', 'val', 'test']}
P = {k: set(zip(out[k].cdr3, out[k]['antigen.epitope'])) for k in ['train', 'val', 'test']}
ok = True
import itertools
for a, b in itertools.combinations(['train', 'val', 'test'], 2):
    eo, co, po = len(E[a] & E[b]), len(C[a] & C[b]), len(P[a] & P[b])
    print(f"  {a:>5} ∩ {b:<5}  epitopes {eo:>4}   CLUSTERS {co:>4}   exact pairs {po:>4}")
    if eo or co or po:
        ok = False
print(f"\n  cross-boundary clusters: {'ZERO' if ok else '*** NON-ZERO ***'}")

print("\n  minimum edit distance from each TEST epitope to the nearest TRAIN epitope:")
tr = sorted(E['train'])
mins = []
for e in sorted(E['test']):
    best = min(Levenshtein.distance(e, t) for t in tr)
    mins.append(best)
mins = np.array(mins)
print(f"    min {mins.min()}  |  distance 1: {int((mins == 1).sum())}  "
      f"2: {int((mins == 2).sum())}  3: {int((mins == 3).sum())}  "
      f">=4: {int((mins >= 4).sum())}")
idmax = []
for e in sorted(E['test']):
    idmax.append(max(1 - Levenshtein.distance(e, t) / max(len(e), len(t)) for t in tr))
idmax = np.array(idmax)
print(f"    max identity to any training epitope: {idmax.max():.3f} "
      f"(>= {MIN_IDENT:.0%}: {int((idmax >= MIN_IDENT).sum())} epitopes)")
clean = bool(mins.min() > MAX_DIST and idmax.max() < MIN_IDENT)
print(f"    separation rule holds on the realised split: {clean}")

OUT.mkdir(exist_ok=True)
for k, d in out.items():
    d[['cdr3', 'antigen.epitope', 'label']].to_csv(OUT / f'{k}.tsv', sep='\t', index=False)
json.dump(dict(seed=SEED, max_dist=MAX_DIST, min_identity=MIN_IDENT,
               n_epitopes=n, n_clusters=len(clusters),
               cluster_sizes=dict(singleton=int((sizes == 1).sum()), two=int((sizes == 2).sum()),
                                  three_to_five=int(sizes.between(3, 5).sum()),
                                  six_to_ten=int(sizes.between(6, 10).sum()),
                                  gt_ten=int((sizes > 10).sum()),
                                  max=int(sizes.max()), median=float(sizes.median()),
                                  mean=float(sizes.mean())),
               largest_clusters=[sorted(c) for c in clusters[:5]],
               splits={k: dict(rows=int(len(d)), epitopes=int(d['antigen.epitope'].nunique()),
                               cdr3=int(d.cdr3.nunique()), pos=int(d.label.sum()))
                       for k, d in out.items()},
               cross_boundary_clusters=0 if ok else -1,
               test_min_dist_to_train=dict(min=int(mins.min()),
                                           d1=int((mins == 1).sum()), d2=int((mins == 2).sum()),
                                           d3=int((mins == 3).sum()), d4plus=int((mins >= 4).sum())),
               test_max_identity_to_train=float(idmax.max()),
               separation_holds=clean),
          open(OUT / 'split_report.json', 'w'), indent=2)
print(f"\n  written to {OUT}/  (train, val, test, test_balanced .tsv + split_report.json)")
