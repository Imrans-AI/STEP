"""
Length-based masking for the physicochemical interaction map (reviewer R1-2).

The maps are 20 x 15 but real pairs are shorter, so cells beyond a pair's actual
CDR3b / epitope length are padding. They are currently left in place and multiplied
by the structure prior.

Lengths are taken from the pair key in data/embeddings/physchem_index.json, whose
keys are "<cdr3>|<epitope>" - i.e. from the sequences themselves. Padding is NEVER
inferred by testing map values for zero: valid cells are frequently zero too
(abs(prop_a - prop_b) == 0 whenever the two residues share a property value, which
for the charge channel is most of the map), so a zero test would mask real signal.

Nothing here changes behaviour on its own; STEPModel applies it only when
use_length_mask=True.
"""
import json
from pathlib import Path

import numpy as np
import torch

MAX_CDR3, MAX_EPI = 20, 15

_DEFAULT_INDEX = Path(__file__).parent.parent / 'data' / 'embeddings' / 'physchem_index.json'


def load_length_table(index_path=None):
    """idx -> (cdr3_len, epi_len), clipped to the map window, from physchem_index.json.

    Returns two int64 arrays indexed by the same integer used to address
    physchem_maps.npy, so a dataset can look lengths up by map index alone.
    """
    index_path = Path(index_path) if index_path is not None else _DEFAULT_INDEX
    with open(index_path, 'r') as f:
        physchem_index = json.load(f)

    n = max(physchem_index.values()) + 1
    cdr3_len = np.zeros(n, dtype=np.int64)
    epi_len = np.zeros(n, dtype=np.int64)

    for pair_key, idx in physchem_index.items():
        cdr3, epitope = pair_key.split('|')
        cdr3_len[idx] = min(len(cdr3), MAX_CDR3)
        epi_len[idx] = min(len(epitope), MAX_EPI)

    return cdr3_len, epi_len


def lengths_from_key(pair_key):
    """(cdr3_len, epi_len) for a single "<cdr3>|<epitope>" key, clipped to the window."""
    cdr3, epitope = pair_key.split('|')
    return min(len(cdr3), MAX_CDR3), min(len(epitope), MAX_EPI)


def build_mask(cdr3_len, epi_len, height=MAX_CDR3, width=MAX_EPI, device=None, dtype=torch.float32):
    """(B, 1, height, width) mask: 1.0 inside a pair's real extent, 0.0 on padding.

    cdr3_len / epi_len are per-sample lengths, shape (B,). The singleton channel
    axis broadcasts the same mask across all 5 physicochemical channels.
    """
    cdr3_len = torch.as_tensor(cdr3_len, device=device).reshape(-1)
    epi_len = torch.as_tensor(epi_len, device=device).reshape(-1)
    if cdr3_len.shape != epi_len.shape:
        raise ValueError(
            f"cdr3_len and epi_len must have the same shape, got "
            f"{tuple(cdr3_len.shape)} and {tuple(epi_len.shape)}"
        )

    rows = torch.arange(height, device=cdr3_len.device).unsqueeze(0) < cdr3_len.unsqueeze(1)
    cols = torch.arange(width, device=epi_len.device).unsqueeze(0) < epi_len.unsqueeze(1)
    mask = rows.unsqueeze(2) & cols.unsqueeze(1)          # (B, height, width)
    return mask.unsqueeze(1).to(dtype)                    # (B, 1, height, width)
