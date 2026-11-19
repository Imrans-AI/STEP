"""
CREATE_SPLIT_v2.py — normalised rebuild of the 600-epitope split (reviewer R2-1 / R2-2).

Based on CREATE_SMALL_EPITOPE_SPLIT.py, with the same design and the same seed:
  600 most frequent epitopes -> train, 100 -> validation, remainder -> test,
  epitope-disjoint, plus a balanced test subset (all positives + an equal number
  of negatives, seed 42), as in CREATE_BALANCED_CLEAN_SPLIT.py.

What is new: sequences are NORMALISED on read, before anything else.

  Why: the released splits compare raw strings when asserting disjointness, so
  'ELAGIGILTV\\xa0' (non-breaking space, from UniPMT) in test and 'ELAGIGILTV' in
  train counted as two different epitopes. After normalisation the released splits
  show train/test exact-pair overlap 4 and epitope overlap 1, plus val/test
  overlap 1. See revision/evidence/R2-1_whitespace_contamination.md.

Writes data/splits_600_epitope_v2/. The originals in data/splits_600_epitope/ and
data/embeddings/ are never opened for writing by this script.

Run from the data/ directory:  python CREATE_SPLIT_v2.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
POOL = HERE / 'splits_true_unseen'
ORIG = HERE / 'splits_600_epitope'
OUT = HERE / 'splits_600_epitope_v2'

SEED = 42
N_TRAIN_EPI = 600
N_VAL_EPI = 100
EPI_MIN, EPI_MAX = 8, 15          # the Methods' stated epitope length range
AA = set('ACDEFGHIKLMNPQRSTVWY')
ZERO_WIDTH = ['​', '‌', '‍', '﻿']

sep = "=" * 78


def normalise(s):
    """strip whitespace, remove NBSP and zero-width characters, uppercase."""
    if not isinstance(s, str):
        return s
    s = s.replace('\xa0', '')
    for z in ZERO_WIDTH:
        s = s.replace(z, '')
    return s.strip().upper()


def strip_ws(s):
    """whitespace/zero-width removal only — no case change, for attributing repairs."""
    if not isinstance(s, str):
        return s
    s = s.replace('\xa0', '')
    for z in ZERO_WIDTH:
        s = s.replace(z, '')
    return s.strip()


def repair(s):
    """Repair a normalised sequence: drop a trailing ';' terminator.

    Case is already repaired by normalise(). Anything still outside the 20 standard
    residues after this (e.g. '*', 'O', 'U', 'X', digits, internal spaces) is not
    guessable and the row is dropped instead.
    """
    if not isinstance(s, str):
        return s
    return s.rstrip(';')


def is_clean(s):
    return isinstance(s, str) and len(s) > 0 and all(c in AA for c in s)


def load_pool():
    print(sep); print("  1. POOL"); print(sep)
    a = pd.read_csv(POOL / 'train.tsv', sep='\t')
    b = pd.read_csv(POOL / 'test_true_unseen_epitope.tsv', sep='\t')
    df = pd.concat([a, b], ignore_index=True)
    print(f"  {POOL.name}/train.tsv                      {len(a):>7,} rows")
    print(f"  {POOL.name}/test_true_unseen_epitope.tsv   {len(b):>7,} rows")
    print(f"  pooled                                      {len(df):>7,} rows")
    print(f"  epitopes (raw strings): {df['antigen.epitope'].nunique()}")
    return df


def clean(df):
    print("\n" + sep); print("  2. NORMALISATION AND FILTERING"); print(sep)
    n0 = len(df)
    e0, c0 = df['antigen.epitope'].nunique(), df['cdr3'].nunique()

    df = df.copy()
    df['cdr3_raw'] = df['cdr3']
    df['antigen.epitope_raw'] = df['antigen.epitope']
    df['cdr3'] = df['cdr3'].map(normalise)
    df['antigen.epitope'] = df['antigen.epitope'].map(normalise)
    e1, c1 = df['antigen.epitope'].nunique(), df['cdr3'].nunique()
    print(f"  normalisation merged epitopes : {e0} -> {e1}  ({e0 - e1} were duplicate spellings)")
    print(f"  normalisation merged CDR3b    : {c0:,} -> {c1:,}  ({c0 - c1:,} duplicate spellings)")

    # --- REPAIR, then drop only what is still unusable ------------------------
    print("\n  REPAIRS (applied before any row is dropped):")
    for col, label in [('cdr3', 'CDR3b'), ('antigen.epitope', 'epitope')]:
        raw_ws = df[col + '_raw'].map(strip_ws)
        cased = (raw_ws != df[col]) & (raw_ws.map(lambda x: x.upper() if isinstance(x, str) else x) == df[col])
        print(f"    {label:8s} case-normalised (lowercase residues upper-cased): "
              f"{int(cased.sum()):>6,} rows, {raw_ws[cased].nunique()} distinct values")

    semi = {}
    for col, label in [('cdr3', 'CDR3b'), ('antigen.epitope', 'epitope')]:
        rep = df[col].map(repair)
        changed = rep != df[col]
        semi[label] = (int(changed.sum()), df.loc[changed, col].nunique())
        df[col] = rep
    for label, (n, d) in semi.items():
        print(f"    {label:8s} trailing ';' stripped: {n:>6,} rows, {d} distinct values")
    total_repaired = sum(n for n, _ in semi.values())

    ok_e = df['antigen.epitope'].map(is_clean)
    ok_c = df['cdr3'].map(is_clean)
    bad_e_vals = sorted(set(df.loc[~ok_e, 'antigen.epitope'].dropna()))
    bad_c_vals = sorted(set(df.loc[~ok_c, 'cdr3'].dropna()))
    print(f"\n  DROPPED — still outside the 20 standard residues after repair:")
    print(f"    epitope side : {int((~ok_e).sum()):>7,} rows  ({len(bad_e_vals)} distinct values)")
    print(f"    CDR3b side   : {int((~ok_c).sum()):>7,} rows  ({len(bad_c_vals)} distinct values)")
    if bad_e_vals:
        print(f"      epitopes: {[repr(x) for x in bad_e_vals[:8]]}")
    if bad_c_vals:
        print(f"      CDR3b   : {[repr(x) for x in bad_c_vals[:8]]}")
        resid = sorted({c for v in bad_c_vals for c in v if c not in AA})
        print(f"      offending characters: {resid}")
    df = df[ok_e & ok_c]
    print(f"\n    repaired (kept) {total_repaired:>7,} rows")
    print(f"    dropped         {int((~(ok_e & ok_c)).sum()):>7,} rows")
    print(f"    remaining       {len(df):>7,} rows")

    # --- epitope length filter ----------------------------------------------
    L = df['antigen.epitope'].str.len()
    keep = L.between(EPI_MIN, EPI_MAX)
    removed = df[~keep]
    print(f"\n  epitope length filter {EPI_MIN}-{EPI_MAX} aa:")
    print(f"    rows removed   {len(removed):>7,}  "
          f"({removed['antigen.epitope'].nunique()} distinct epitopes)")
    if len(removed):
        for e, n in removed['antigen.epitope'].value_counts().items():
            print(f"      {e!r:20s} len {len(e):>2d}  {n} rows  "
                  f"({int((removed[removed['antigen.epitope'] == e].label == 1).sum())} pos)")
    df = df[keep]
    print(f"    remaining      {len(df):>7,} rows")

    cl = df['cdr3'].str.len()
    print(f"\n  CDR3b length after cleaning: min {cl.min()} max {cl.max()} "
          f"(outside 8-20: {int((~cl.between(8, 20)).sum())} rows — NOT filtered, "
          f"only the epitope filter was requested)")

    # --- duplicates and label conflicts --------------------------------------
    key = ['cdr3', 'antigen.epitope']
    dup = len(df) - len(df.drop_duplicates(subset=key + ['label']))
    df = df.drop_duplicates(subset=key + ['label'])
    g = df.groupby(key)['label'].nunique()
    conflicted = g[g > 1]
    print(f"\n  exact duplicate rows removed (same cdr3, epitope, label): {dup:,}")
    print(f"  pairs carrying BOTH labels after normalisation: {len(conflicted):,}")
    if len(conflicted):
        print("    these are dropped entirely — a pair cannot be both:")
        for (c, e) in list(conflicted.index)[:8]:
            print(f"      ({c}, {e})")
        if len(conflicted) > 8:
            print(f"      … and {len(conflicted) - 8} more")
        ci = set(conflicted.index)
        before = len(df)
        df = df[~pd.Series(list(zip(df.cdr3, df['antigen.epitope'])), index=df.index).isin(ci)]
        print(f"    rows dropped: {before - len(df):,}")

    df = df.drop(columns=['cdr3_raw', 'antigen.epitope_raw'])
    print(f"\n  FINAL POOL: {len(df):,} rows, {df['antigen.epitope'].nunique()} epitopes, "
          f"{df['cdr3'].nunique():,} CDR3b, {df.label.mean():.1%} positive")
    return df.reset_index(drop=True)


def split(df):
    print("\n" + sep); print("  3. SPLIT (same design, same seed)"); print(sep)
    counts = df.groupby('antigen.epitope').size().sort_values(ascending=False)
    np.random.seed(SEED)
    all_epitopes = df['antigen.epitope'].unique()
    top = counts.head(N_TRAIN_EPI).index.tolist()
    remaining = [e for e in all_epitopes if e not in top]
    np.random.shuffle(remaining)
    val_e = set(remaining[:N_VAL_EPI])
    test_e = set(remaining[N_VAL_EPI:])
    train_e = set(top)
    print(f"  train epitopes {len(train_e)} | val {len(val_e)} | test {len(test_e)}"
          f" | total {len(train_e) + len(val_e) + len(test_e)}")

    out = {
        'train': df[df['antigen.epitope'].isin(train_e)],
        'val': df[df['antigen.epitope'].isin(val_e)],
        'test': df[df['antigen.epitope'].isin(test_e)],
    }
    pos = out['test'][out['test'].label == 1]
    neg = out['test'][out['test'].label == 0]
    np.random.seed(SEED)
    nb = neg.sample(n=min(len(pos), len(neg)), random_state=SEED)
    bal = pd.concat([pos, nb], ignore_index=True).sample(frac=1, random_state=SEED)
    out['test_balanced'] = bal.reset_index(drop=True)
    return out


def report(out):
    print("\n" + sep); print("  4. PER-SPLIT SUMMARY"); print(sep)
    print(f"  {'split':16s} {'rows':>9s} {'epitopes':>9s} {'CDR3b':>9s} {'pos':>8s} {'pos frac':>9s}")
    for k, d in out.items():
        print(f"  {k:16s} {len(d):>9,} {d['antigen.epitope'].nunique():>9,} "
              f"{d['cdr3'].nunique():>9,} {int(d.label.sum()):>8,} {d.label.mean():>8.1%}")

    print("\n" + sep); print("  5. OVERLAPS ON NORMALISED STRINGS"); print(sep)
    import itertools
    E = {k: set(d['antigen.epitope']) for k, d in out.items()}
    P = {k: set(zip(d.cdr3, d['antigen.epitope'])) for k, d in out.items()}
    C = {k: set(d.cdr3) for k, d in out.items()}
    print(f"  {'pair':34s} {'epitopes':>9s} {'pairs':>9s} {'CDR3b':>9s}")
    allzero = True
    for a, b in itertools.combinations(out, 2):
        eo, po, co = len(E[a] & E[b]), len(P[a] & P[b]), len(C[a] & C[b])
        tag = ''
        if {a, b} != {'test', 'test_balanced'}:
            if eo: allzero = False; tag = '  <-- NON-ZERO'
        print(f"  {a + ' ∩ ' + b:32s} {eo:>9,} {po:>9,} {co:>9,}{tag}")
    print()
    if allzero:
        print("  ALL EPITOPE OVERLAPS BETWEEN train / val / test ARE ZERO ON NORMALISED STRINGS.")
        print("  (test ∩ test_balanced is non-zero by construction: test_balanced is a subset.)")
    else:
        print("  *** AN EPITOPE OVERLAP IS NON-ZERO — SPLIT IS NOT CLEAN ***")
    return allzero


def before_after(out, prev_dir=None):
    print("\n" + sep); print("  6. BEFORE / AFTER vs data/splits_600_epitope/"); print(sep)
    old = {}
    for k in ['train', 'val', 'test', 'test_balanced']:
        p = ORIG / f'{k}.tsv'
        if p.exists():
            old[k] = pd.read_csv(p, sep='\t')

    if prev_dir is not None and Path(prev_dir).exists():
        prev = {}
        for k in ['train', 'val', 'test', 'test_balanced']:
            p = Path(prev_dir) / f'{k}.tsv'
            if p.exists():
                prev[k] = pd.read_csv(p, sep='\t')
        if prev:
            print("  vs the PREVIOUS v2 (drop-only, no repair):")
            print(f"  {'split':15s} {'rows prev':>10s} {'now':>9s} {'Δ':>8s}   "
                  f"{'epi prev':>9s} {'now':>5s}   {'CDR3b prev':>11s} {'now':>9s}   "
                  f"{'pos% prev':>10s} {'now':>7s}")
            for k in out:
                if k not in prev:
                    continue
                o, n = prev[k], out[k]
                print(f"  {k:15s} {len(o):>10,} {len(n):>9,} {len(n) - len(o):>+8,}   "
                      f"{o['antigen.epitope'].nunique():>9,} {n['antigen.epitope'].nunique():>5,}   "
                      f"{o['cdr3'].nunique():>11,} {n['cdr3'].nunique():>9,}   "
                      f"{o.label.mean():>9.1%} {n.label.mean():>6.1%}")
            print()
    print("  vs the ORIGINAL splits:")
    print(f"  {'split':15s} {'rows old':>10s} {'new':>9s} {'Δ':>8s}   "
          f"{'epi old':>8s} {'new':>6s}   {'CDR3b old':>10s} {'new':>9s}   {'pos% old':>9s} {'new':>7s}")
    for k in out:
        if k not in old:
            continue
        o, n = old[k], out[k]
        print(f"  {k:15s} {len(o):>10,} {len(n):>9,} {len(n) - len(o):>+8,}   "
              f"{o['antigen.epitope'].nunique():>8,} {n['antigen.epitope'].nunique():>6,}   "
              f"{o['cdr3'].nunique():>10,} {n['cdr3'].nunique():>9,}   "
              f"{o.label.mean():>8.1%} {n.label.mean():>6.1%}")

    def nz(s):
        return s.replace('\xa0', '').strip().upper()
    print("\n  Overlaps, ORIGINAL splits recomputed on normalised strings (for contrast):")
    import itertools
    EO = {k: {nz(x) for x in d['antigen.epitope']} for k, d in old.items()}
    PO = {k: {(nz(c), nz(e)) for c, e in zip(d.cdr3, d['antigen.epitope'])} for k, d in old.items()}
    EN = {k: set(d['antigen.epitope']) for k, d in out.items()}
    PN = {k: set(zip(d.cdr3, d['antigen.epitope'])) for k, d in out.items()}
    print(f"  {'pair':30s} {'epi orig':>9s} {'epi v2':>7s}   {'pair orig':>10s} {'pair v2':>8s}")
    for a, b in itertools.combinations(['train', 'val', 'test'], 2):
        print(f"  {a + ' ∩ ' + b:28s} {len(EO[a] & EO[b]):>9,} {len(EN[a] & EN[b]):>7,}   "
              f"{len(PO[a] & PO[b]):>10,} {len(PN[a] & PN[b]):>8,}")


def save(out):
    print("\n" + sep); print("  7. WRITE"); print(sep)
    OUT.mkdir(exist_ok=True)
    for k, d in out.items():
        p = OUT / f'{k}.tsv'
        d.to_csv(p, sep='\t', index=False)
        print(f"  wrote {p}  ({len(d):,} rows)")
    print(f"\n  originals untouched: {ORIG}/ not opened for writing")


if __name__ == '__main__':
    pool = load_pool()
    cleaned = clean(pool)
    out = split(cleaned)
    ok = report(out)
    import sys
    prev = sys.argv[1] if len(sys.argv) > 1 else None
    before_after(out, prev)
    if ok:
        save(out)
    else:
        print("\n  NOT WRITING: an epitope overlap was non-zero.")
