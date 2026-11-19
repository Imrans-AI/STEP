"""
Structure-prior control variants for reviewer comments R1-2 / R2-4.

R2-4 asks for shuffled, uniform and length-only mask controls. Each variant
returns (weights, use_length_mask):

  corrected    structure_priors_v2.npz as built by data/build_structure_priors_v2.py
  none         all ones — no structural information at all
  length_only  all ones + per-pair length masking, i.e. the prior encodes ONLY
               which cells lie inside a pair's real sequence extent and carries no
               contact information
  shuffled     the corrected weights with their 300 cell values randomly permuted,
               destroying the spatial pattern while preserving the value histogram
  uniform      every cell at the mean weight of the corrected prior over COVERED
               cells, preserving the overall scale and nothing else

Note on `length_only` vs `none`: physchem_maps.npy is already exactly zero beyond
each pair's extent (see revision/evidence/R1-2_padding_audit.md), so multiplying by
a length mask removes cells that are already zero. The two controls are therefore
expected to be numerically equivalent; both are run so that this is demonstrated
rather than asserted.
"""
from pathlib import Path

import numpy as np

MAX_CDR3, MAX_EPI = 20, 15
EMB = Path(__file__).parent.parent / 'data' / 'embeddings'
VARIANTS = ['corrected', 'none', 'length_only', 'shuffled', 'uniform']


def get_prior(variant, seed=42):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    z = np.load(EMB / 'structure_priors_v2.npz')
    W = z['contact_weights'].astype(np.float32)

    if variant == 'corrected':
        return W, False
    if variant == 'none':
        return np.ones((MAX_CDR3, MAX_EPI), dtype=np.float32), False
    if variant == 'length_only':
        return np.ones((MAX_CDR3, MAX_EPI), dtype=np.float32), True
    if variant == 'shuffled':
        rng = np.random.default_rng(seed)
        flat = W.flatten().copy()
        rng.shuffle(flat)
        return flat.reshape(W.shape).astype(np.float32), False
    if variant == 'uniform':
        covered = z['coverage'] > 0
        return np.full((MAX_CDR3, MAX_EPI), float(W[covered].mean()), dtype=np.float32), False


def describe(variant, seed=42):
    W, mask = get_prior(variant, seed)
    return (f"{variant:12s} mean={W.mean():.4f} min={W.min():.4f} max={W.max():.4f} "
            f"std={W.std():.4f} length_mask={mask}")


if __name__ == '__main__':
    for v in VARIANTS:
        print(describe(v, 42))
    print("\nshuffled differs per seed:")
    for s in (42, 43, 44):
        W, _ = get_prior('shuffled', s)
        print(f"  seed {s}: W[0,0]={W[0,0]:.4f} W[5,6]={W[5,6]:.4f} "
              f"argmax={np.unravel_index(W.argmax(), W.shape)}")
