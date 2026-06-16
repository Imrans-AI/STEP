# structure_aware_physchem.py
import numpy as np
import pickle
from pathlib import Path

print("=" * 60)
print("  STRUCTURE-AWARE PHYSICOCHEMICAL FEATURES")
print("=" * 60)

print("\n📦 Loading STCRDab contact matrices...")
with open('raw/stcrdab_cdr3_contacts.pkl', 'rb') as f:
    contact_data = pickle.load(f)
print(f"   Loaded {len(contact_data)} structures")

print("\n🔬 Analyzing contact patterns...")
all_contacts = []
valid_structures = 0

for data in contact_data:
    if 'cdr3_epitope_distance' in data and data['cdr3_epitope_distance'] is not None:
        dist_matrix = data['cdr3_epitope_distance']
        if isinstance(dist_matrix, np.ndarray) and len(dist_matrix.shape) == 2:
            h, w = dist_matrix.shape
            padded = np.zeros((20, 15))
            padded[:min(h, 20), :min(w, 15)] = dist_matrix[:min(h, 20), :min(w, 15)]
            
            # Convert distance to contact (<6Å threshold)
            contact_matrix = (padded < 6.0).astype(float)
            all_contacts.append(contact_matrix)
            valid_structures += 1

print(f"   Valid structures: {valid_structures}")

if len(all_contacts) == 0:
    print("   ⚠️  No valid contact matrices, using uniform weights")
    avg_contacts = np.ones((20, 15)) * 0.5
    contact_weights = np.ones((20, 15))
else:
    # Average contact frequency (0-1 range, 1 = always in contact)
    avg_contacts = np.mean(all_contacts, axis=0)
    print(f"   Average contact matrix shape: {avg_contacts.shape}")
    print(f"   Mean contact frequency: {avg_contacts.mean():.3f}")
    print(f"   Max contact frequency: {avg_contacts.max():.3f}")
    
    # Use contact frequency directly as weights (no aggressive normalization)
    # Just ensure non-zero baseline
    contact_weights = avg_contacts + 0.1  # Add small baseline
    contact_weights = contact_weights / contact_weights.max()  # Normalize to [0,1]

print(f"\n📊 Contact weight statistics:")
print(f"   Mean: {contact_weights.mean():.4f}")
print(f"   Std:  {contact_weights.std():.4f}")
print(f"   Min:  {contact_weights.min():.4f}")
print(f"   Max:  {contact_weights.max():.4f}")

# Find high-contact hotspots
hotspot_threshold = np.percentile(contact_weights, 90)
hotspots = np.where(contact_weights >= hotspot_threshold)
print(f"\n🔥 High-contact hotspots (top 10%):")
print(f"   Threshold: {hotspot_threshold:.4f}")
print(f"   Number of positions: {len(hotspots[0])}")

# Show position distribution
print(f"\n📍 Contact distribution:")
for thresh in [0.3, 0.5, 0.7, 0.9]:
    n = (contact_weights > thresh).sum()
    print(f"   Weights > {thresh}: {n} positions ({n/300*100:.1f}%)")

# Save
output = {
    'contact_weights': contact_weights.astype(np.float32),
    'avg_contacts': avg_contacts.astype(np.float32),
    'n_structures': valid_structures,
    'hotspot_threshold': float(hotspot_threshold)
}

np.savez('embeddings/structure_priors.npz', **output)

print(f"\n✅ Saved embeddings/structure_priors.npz")
print(f"   contact_weights: (20, 15) from {valid_structures} PDB structures")
print(f"\n💡 KEY NOVELTY: Real 3D contact patterns guide physchem learning")
