# build_structure_priors_v2.py
#
# Corrected contact prior for reviewer comment R1-2 / R2-4.
#
# Defect in data/structure_aware_physchem.py (the released path):
#   padded = np.zeros((20, 15))
#   padded[:h, :w] = dist_matrix[...]
#   contact_matrix = (padded < 6.0)      <-- padded zeros are 0.0 A, so they pass
# Every padded cell was counted as a contact in all 198 structures, the mean was
# taken over 198 regardless of coverage, and the +0.1 baseline was normalised by a
# maximum (1.1) that the padding itself set.
#
# This script:
#   1. thresholds over VALID residues only, then places the result into the 20x15 map;
#   2. accumulates a per-cell coverage count (structures that actually reach the cell);
#   3. divides contacts by coverage, not by 198 (uncovered cells -> frequency 0);
#   4. applies the same +0.1 baseline and normalises by the maximum over COVERED cells.
#
# Writes data/embeddings/structure_priors_v2.npz. Does NOT touch structure_priors.npz.
# Read-only with respect to every other artifact; no model or checkpoint is loaded.
#
# Run from the data/ directory:  python build_structure_priors_v2.py

import numpy as np
import pickle
from pathlib import Path

MAX_CDR3, MAX_EPI = 20, 15
CONTACT_THRESHOLD = 6.0
BASELINE = 0.1

HERE = Path(__file__).parent
CONTACTS_PKL = HERE / 'raw' / 'stcrdab_cdr3_contacts.pkl'
OLD_NPZ = HERE / 'embeddings' / 'structure_priors.npz'
OUT_NPZ = HERE / 'embeddings' / 'structure_priors_v2.npz'

sep = "=" * 78


def build():
    print(sep)
    print("  STRUCTURE PRIORS v2 - coverage-aware, threshold before padding")
    print(sep)

    with open(CONTACTS_PKL, 'rb') as f:
        contact_data = pickle.load(f)
    print(f"\nLoaded {len(contact_data)} structures from {CONTACTS_PKL.name}")

    contact_counts = np.zeros((MAX_CDR3, MAX_EPI), dtype=np.float64)
    coverage = np.zeros((MAX_CDR3, MAX_EPI), dtype=np.float64)
    valid_structures = 0
    cdr3_lengths, epi_lengths = [], []
    truncated = 0

    for data in contact_data:
        if 'cdr3_epitope_distance' not in data or data['cdr3_epitope_distance'] is None:
            continue
        dist_matrix = data['cdr3_epitope_distance']
        if not (isinstance(dist_matrix, np.ndarray) and dist_matrix.ndim == 2):
            continue

        h, w = dist_matrix.shape
        hh, ww = min(h, MAX_CDR3), min(w, MAX_EPI)
        if h > MAX_CDR3 or w > MAX_EPI:
            truncated += 1

        # --- THRESHOLD FIRST, over valid residues only -----------------------
        valid_block = dist_matrix[:hh, :ww]
        contact_block = (valid_block < CONTACT_THRESHOLD).astype(np.float64)

        # --- then place into the 20x15 map -----------------------------------
        contact_counts[:hh, :ww] += contact_block
        coverage[:hh, :ww] += 1.0

        valid_structures += 1
        cdr3_lengths.append(hh)
        epi_lengths.append(ww)

    print(f"Valid structures: {valid_structures}")
    print(f"Structures truncated by the 20x15 window: {truncated}")
    print(f"CDR3b length range: {min(cdr3_lengths)}-{max(cdr3_lengths)}")
    print(f"Epitope length range: {min(epi_lengths)}-{max(epi_lengths)}")

    # --- coverage-normalised frequency ---------------------------------------
    covered = coverage > 0
    n_uncovered = int((~covered).sum())
    freq = np.zeros((MAX_CDR3, MAX_EPI), dtype=np.float64)
    freq[covered] = contact_counts[covered] / coverage[covered]

    print(f"\nCells covered by at least one structure: {int(covered.sum())} / 300")
    print(f"Cells covered by NO structure (frequency set to 0): {n_uncovered}")
    if n_uncovered:
        print("  uncovered cells:",
              [tuple(map(int, c)) for c in np.argwhere(~covered)])
    print(f"Minimum coverage over covered cells: {int(coverage[covered].min())}")
    print(f"Maximum coverage: {int(coverage.max())}")
    print(f"Total true contacts (valid cells only): {int(contact_counts.sum())}")

    # --- baseline + normalisation over COVERED cells only ---------------------
    weights = freq + BASELINE
    denominator = float(weights[covered].max())
    peak_cell = np.unravel_index(np.argmax(np.where(covered, weights, -1)), weights.shape)

    print(f"\nNormalisation denominator (max over covered cells): {denominator:.6f}")
    print(f"  driven by cell {tuple(map(int, peak_cell))}: "
          f"frequency {freq[peak_cell]:.6f} "
          f"({int(contact_counts[peak_cell])} contacts / {int(coverage[peak_cell])} covering structures)")
    print(f"  old released denominator was 1.100000 (0.1 + 1.0 from padded cells)")
    print(f"  ratio old/new: {1.1 / denominator:.4f}x")

    contact_weights = (weights / denominator).astype(np.float32)
    avg_contacts = freq.astype(np.float32)

    hotspot_threshold = float(np.percentile(contact_weights, 90))

    output = {
        'contact_weights': contact_weights,
        'avg_contacts': avg_contacts,
        'n_structures': valid_structures,
        'hotspot_threshold': hotspot_threshold,
        # extras specific to v2, additive - the four keys above keep their meaning
        'coverage': coverage.astype(np.float32),
        'contact_counts': contact_counts.astype(np.float32),
        'normalisation_denominator': np.float32(denominator),
    }
    np.savez(OUT_NPZ, **output)
    print(f"\nSaved {OUT_NPZ}")
    print(f"  keys: {sorted(output)}")

    return contact_weights, coverage, freq, denominator


def interface_stats(W, covered, label):
    """Summary statistics for one 20x15 weight matrix."""
    core = np.zeros_like(W, dtype=bool)
    core[4:8, 3:8] = True                      # where true contacts concentrate
    region = np.zeros_like(W, dtype=bool)
    region[0:9, 0:9] = True                    # the block both reviewers name
    peak = np.unravel_index(W.argmax(), W.shape)

    bg_core = covered & ~core
    bg_region = covered & ~region

    print(f"\n--- {label} ---")
    print(f"  max                                   : {W.max():.4f}")
    print(f"  mean over full map                    : {W.mean():.4f}")
    print(f"  mean over CDR3b 0-8 x epitope 0-8     : {W[0:9, 0:9].mean():.4f}")
    print(f"  peak cell                             : {tuple(map(int, peak))} = {W[peak]:.4f}")
    print(f"  mean core (rows 4-7 x cols 3-7)       : {W[core].mean():.4f}")
    print(f"  mean background (covered, outside core): {W[bg_core].mean():.4f}")
    print(f"  ENRICHMENT core / background           : {W[core].mean() / W[bg_core].mean():.4f}x")
    print(f"  mean region (rows 0-8 x cols 0-8)     : {W[region].mean():.4f}")
    print(f"  mean background (covered, outside region): {W[bg_region].mean():.4f}")
    print(f"  ENRICHMENT region / background         : {W[region].mean() / W[bg_region].mean():.4f}x")
    print(f"  cells equal to 1.0                    : {int((W >= 0.99999).sum())}")
    print(f"  min                                   : {W.min():.4f}")


def side_by_side(old, new):
    print("\n" + sep)
    print("  OLD (structure_priors.npz)                      |  NEW (structure_priors_v2.npz)")
    print(sep)
    hdr = " ".join(f"{j:>5d}" for j in range(MAX_EPI))
    print(f"      {hdr}   |       {hdr}")
    for i in range(MAX_CDR3):
        l = " ".join(f"{v:5.3f}" for v in old[i])
        r = " ".join(f"{v:5.3f}" for v in new[i])
        print(f"r{i:>2d}: {l}   | r{i:>2d}: {r}")


if __name__ == '__main__':
    new_W, coverage, freq, denom = build()
    old_W = np.load(OLD_NPZ)['contact_weights']
    covered = coverage > 0

    side_by_side(old_W, new_W)

    print("\n" + sep)
    print("  COMPARISON")
    print(sep)
    interface_stats(old_W, covered, "OLD  structure_priors.npz")
    interface_stats(new_W, covered, "NEW  structure_priors_v2.npz")

    print(f"\n  max |new - old| over the full map   : {np.abs(new_W - old_W).max():.4f}")
    print(f"  mean |new - old| over the full map  : {np.abs(new_W - old_W).mean():.4f}")
    print(f"  old is unchanged on disk            : "
          f"{np.array_equal(old_W, np.load(OLD_NPZ)['contact_weights'])}")

    # diagnostic only, not saved: how much of the new peak rests on thin coverage
    print("\n  Coverage behind the 10 highest new weights:")
    flat = np.argsort(new_W, axis=None)[::-1][:10]
    for f in flat:
        i, j = np.unravel_index(f, new_W.shape)
        print(f"    ({i:>2d},{j:>2d})  weight {new_W[i, j]:.4f}  "
              f"contacts {int(np.load(OUT_NPZ)['contact_counts'][i, j]):>3d}  "
              f"coverage {int(coverage[i, j]):>3d}")
