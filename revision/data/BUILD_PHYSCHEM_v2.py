"""
BUILD_PHYSCHEM_v2.py — physicochemical maps for the normalised v2 splits (R2-1).

Two jobs:
  1. Audit data/embeddings/physchem_index.json for keys that differ only by
     whitespace / case, i.e. duplicate spellings of the same pair.
  2. Provide a map for every pair in data/splits_600_epitope_v2/ whose normalised
     form has no entry in the existing index, computing only the missing ones.

The original physchem_maps.npy (1.5 GB) and physchem_index.json are the artifacts
the submitted checkpoint was trained against, so this script NEVER writes to them.
Missing maps go to a separate file and the v2 index addresses both:

    idx <  N_ORIG   ->  row idx of physchem_maps.npy          (reused, unchanged)
    idx >= N_ORIG   ->  row (idx - N_ORIG) of physchem_maps_v2_extra.npy

Property tables and the map function are copied verbatim from compute_physchem.py.

Run from the data/ directory:  python BUILD_PHYSCHEM_v2.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EMB = HERE / 'embeddings'
V2 = HERE / 'splits_600_epitope_v2'
OUT_EXTRA = EMB / 'physchem_maps_v2_extra.npy'
OUT_INDEX = EMB / 'physchem_index_v2.json'

MAX_CDR3, MAX_EPI = 20, 15
ZERO_WIDTH = ['​', '‌', '‍', '﻿']

HYDROPHOBICITY = {'A': 0.62, 'R': 0.00, 'N': 0.16, 'D': 0.11, 'C': 0.68,
                  'Q': 0.19, 'E': 0.14, 'G': 0.48, 'H': 0.32, 'I': 1.00,
                  'L': 0.98, 'K': 0.05, 'M': 0.85, 'F': 0.97, 'P': 0.51,
                  'S': 0.35, 'T': 0.39, 'W': 0.89, 'Y': 0.63, 'V': 0.86}
CHARGE = {'A': 0.5, 'R': 1.0, 'N': 0.5, 'D': 0.0, 'C': 0.5,
          'Q': 0.5, 'E': 0.0, 'G': 0.5, 'H': 0.75, 'I': 0.5,
          'L': 0.5, 'K': 1.0, 'M': 0.5, 'F': 0.5, 'P': 0.5,
          'S': 0.5, 'T': 0.5, 'W': 0.5, 'Y': 0.5, 'V': 0.5}
FLEXIBILITY = {'A': 0.35, 'R': 0.53, 'N': 0.46, 'D': 0.51, 'C': 0.35,
               'Q': 0.49, 'E': 0.50, 'G': 0.54, 'H': 0.32, 'I': 0.46,
               'L': 0.37, 'K': 0.47, 'M': 0.30, 'F': 0.31, 'P': 0.51,
               'S': 0.51, 'T': 0.44, 'W': 0.31, 'Y': 0.42, 'V': 0.39}
REFRACTIVITY = {'A': 0.20, 'R': 0.65, 'N': 0.33, 'D': 0.32, 'C': 0.35,
                'Q': 0.46, 'E': 0.44, 'G': 0.00, 'H': 0.58, 'I': 0.57,
                'L': 0.57, 'K': 0.52, 'M': 0.59, 'F': 0.76, 'P': 0.36,
                'S': 0.23, 'T': 0.33, 'W': 1.00, 'Y': 0.78, 'V': 0.48}
AREA_LOSS = {'A': 0.31, 'R': 0.64, 'N': 0.43, 'D': 0.41, 'C': 0.41,
             'Q': 0.51, 'E': 0.49, 'G': 0.00, 'H': 0.59, 'I': 0.68,
             'L': 0.68, 'K': 0.60, 'M': 0.67, 'F': 0.76, 'P': 0.51,
             'S': 0.33, 'T': 0.44, 'W': 1.00, 'Y': 0.82, 'V': 0.61}
PROPERTIES = {'hydrophobicity': HYDROPHOBICITY, 'charge': CHARGE,
              'flexibility': FLEXIBILITY, 'refractivity': REFRACTIVITY,
              'area_loss': AREA_LOSS}


def compute_physchem_map(cdr3, epitope, max_cdr3_len=MAX_CDR3, max_epi_len=MAX_EPI):
    physchem_map = np.zeros((len(PROPERTIES), max_cdr3_len, max_epi_len), dtype=np.float32)
    cdr3_len, epi_len = min(len(cdr3), max_cdr3_len), min(len(epitope), max_epi_len)
    for pi, (_, pd_) in enumerate(PROPERTIES.items()):
        for i in range(cdr3_len):
            for j in range(epi_len):
                a, b = cdr3[i], epitope[j]
                if a in pd_ and b in pd_:
                    physchem_map[pi, i, j] = abs(pd_[a] - pd_[b])
    return physchem_map


def normalise(s):
    s = s.replace('\xa0', '')
    for z in ZERO_WIDTH:
        s = s.replace(z, '')
    return s.strip().upper()


sep = "=" * 78

print(sep); print("  1. AUDIT OF THE EXISTING physchem_index.json"); print(sep)
idx = json.load(open(EMB / 'physchem_index.json'))
n_orig_rows = np.load(EMB / 'physchem_maps.npy', mmap_mode='r').shape[0]
print(f"  index entries      : {len(idx):,}")
print(f"  physchem_maps rows : {n_orig_rows:,}")
print(f"  unindexed rows     : {n_orig_rows - len(idx):,}")

groups = {}
for k in idx:
    c, e = k.split('|', 1)
    groups.setdefault(f"{normalise(c)}|{normalise(e)}", []).append(k)
dups = {k: v for k, v in groups.items() if len(v) > 1}
print(f"\n  distinct keys after normalisation : {len(groups):,}")
print(f"  normalised forms with MORE THAN ONE raw key: {len(dups):,}")
print(f"  redundant keys (duplicate spellings)       : {sum(len(v) - 1 for v in dups.values()):,}")
if dups:
    print("\n  >>> The ORIGINAL index DOES contain duplicate keys differing only by")
    print("      whitespace/case. Examples:")
    for k, v in list(dups.items())[:6]:
        print(f"        normalised {k!r}")
        for raw in v:
            print(f"          raw {raw!r}  idx {idx[raw]}")
    affected = sorted({k.split('|', 1)[1] for k in dups})
    print(f"\n      distinct epitopes involved: {len(affected)} -> {affected}")

print("\n" + sep); print("  2. COVERAGE OF THE v2 SPLITS"); print(sep)
pairs = set()
for f in ['train', 'val', 'test', 'test_balanced']:
    d = pd.read_csv(V2 / f'{f}.tsv', sep='\t')
    pairs |= set(zip(d.cdr3, d['antigen.epitope']))
    print(f"  {f:14s} {len(d):>8,} rows")
print(f"\n  distinct v2 pairs: {len(pairs):,}")

have, missing = {}, []
for c, e in sorted(pairs):
    k = f"{c}|{e}"
    if k in idx:
        have[k] = idx[k]
    else:
        missing.append((c, e))
print(f"  already in physchem_index.json : {len(have):,}")
print(f"  MISSING, need computing        : {len(missing):,}")
if missing[:8]:
    print(f"    e.g. {[f'{c}|{e}' for c, e in missing[:8]]}")

print("\n" + sep); print("  3. COMPUTE THE MISSING MAPS"); print(sep)
v2_index = dict(have)
if missing:
    extra = np.zeros((len(missing), len(PROPERTIES), MAX_CDR3, MAX_EPI), dtype=np.float32)
    for i, (c, e) in enumerate(missing):
        extra[i] = compute_physchem_map(c, e)
        v2_index[f"{c}|{e}"] = n_orig_rows + i
    np.save(OUT_EXTRA, extra)
    print(f"  computed {len(missing):,} maps, shape {extra.shape}")
    print(f"  wrote {OUT_EXTRA}  ({OUT_EXTRA.stat().st_size / 1e6:.2f} MB)")
    print(f"  addressing: idx >= {n_orig_rows:,} -> row (idx - {n_orig_rows:,}) of this file")
else:
    print("  nothing missing; no extra file written")

with open(OUT_INDEX, 'w') as f:
    json.dump(v2_index, f)
print(f"  wrote {OUT_INDEX}  ({len(v2_index):,} keys)")

print("\n" + sep); print("  4. VERIFY THE v2 INDEX"); print(sep)
g2 = {}
for k in v2_index:
    c, e = k.split('|', 1)
    g2.setdefault(f"{normalise(c)}|{normalise(e)}", []).append(k)
d2 = {k: v for k, v in g2.items() if len(v) > 1}
print(f"  v2 index keys                                   : {len(v2_index):,}")
print(f"  distinct after normalisation                    : {len(g2):,}")
print(f"  keys differing only by whitespace/case           : {len(d2):,}")
print(f"  every key already equals its normalised form     : "
      f"{all(k == f'{normalise(k.split(chr(124),1)[0])}|{normalise(k.split(chr(124),1)[1])}' for k in v2_index)}")
print(f"  every v2 pair has a map                          : "
      f"{all(f'{c}|{e}' in v2_index for c, e in pairs)}")

if missing:
    chk = np.load(OUT_EXTRA, mmap_mode='r')
    c, e = missing[0]
    ref = compute_physchem_map(c, e)
    print(f"  spot-check {c}|{e}: stored == recomputed -> "
          f"{np.array_equal(np.asarray(chk[0]), ref)}")

print(f"\n  originals untouched: physchem_maps.npy and physchem_index.json not written")
