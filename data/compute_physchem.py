# compute_physchem.py
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

print("=" * 60)
print("  PHYSICOCHEMICAL MAP COMPUTATION")
print("=" * 60)

# Physicochemical property scales (normalized to [0,1])
HYDROPHOBICITY = {
    'A': 0.62, 'R': 0.00, 'N': 0.16, 'D': 0.11, 'C': 0.68,
    'Q': 0.19, 'E': 0.14, 'G': 0.48, 'H': 0.32, 'I': 1.00,
    'L': 0.98, 'K': 0.05, 'M': 0.85, 'F': 0.97, 'P': 0.51,
    'S': 0.35, 'T': 0.39, 'W': 0.89, 'Y': 0.63, 'V': 0.86
}

CHARGE = {
    'A': 0.5, 'R': 1.0, 'N': 0.5, 'D': 0.0, 'C': 0.5,
    'Q': 0.5, 'E': 0.0, 'G': 0.5, 'H': 0.75, 'I': 0.5,
    'L': 0.5, 'K': 1.0, 'M': 0.5, 'F': 0.5, 'P': 0.5,
    'S': 0.5, 'T': 0.5, 'W': 0.5, 'Y': 0.5, 'V': 0.5
}

FLEXIBILITY = {
    'A': 0.35, 'R': 0.53, 'N': 0.46, 'D': 0.51, 'C': 0.35,
    'Q': 0.49, 'E': 0.50, 'G': 0.54, 'H': 0.32, 'I': 0.46,
    'L': 0.37, 'K': 0.47, 'M': 0.30, 'F': 0.31, 'P': 0.51,
    'S': 0.51, 'T': 0.44, 'W': 0.31, 'Y': 0.42, 'V': 0.39
}

REFRACTIVITY = {
    'A': 0.20, 'R': 0.65, 'N': 0.33, 'D': 0.32, 'C': 0.35,
    'Q': 0.46, 'E': 0.44, 'G': 0.00, 'H': 0.58, 'I': 0.57,
    'L': 0.57, 'K': 0.52, 'M': 0.59, 'F': 0.76, 'P': 0.36,
    'S': 0.23, 'T': 0.33, 'W': 1.00, 'Y': 0.78, 'V': 0.48
}

AREA_LOSS = {
    'A': 0.31, 'R': 0.64, 'N': 0.43, 'D': 0.41, 'C': 0.41,
    'Q': 0.51, 'E': 0.49, 'G': 0.00, 'H': 0.59, 'I': 0.68,
    'L': 0.68, 'K': 0.60, 'M': 0.67, 'F': 0.76, 'P': 0.51,
    'S': 0.33, 'T': 0.44, 'W': 1.00, 'Y': 0.82, 'V': 0.61
}

PROPERTIES = {
    'hydrophobicity': HYDROPHOBICITY,
    'charge': CHARGE,
    'flexibility': FLEXIBILITY,
    'refractivity': REFRACTIVITY,
    'area_loss': AREA_LOSS
}

print(f"   Properties used: {list(PROPERTIES.keys())}")
print(f"   ✅ Properties normalized to [0,1]")

def compute_physchem_map(cdr3, epitope, max_cdr3_len=20, max_epi_len=15):
    """Compute 5-channel physicochemical interaction map"""
    n_props = len(PROPERTIES)
    physchem_map = np.zeros((n_props, max_cdr3_len, max_epi_len), dtype=np.float32)
    
    cdr3_len = min(len(cdr3), max_cdr3_len)
    epi_len = min(len(epitope), max_epi_len)
    
    for prop_idx, (prop_name, prop_dict) in enumerate(PROPERTIES.items()):
        for i in range(cdr3_len):
            for j in range(epi_len):
                aa_cdr3 = cdr3[i]
                aa_epi = epitope[j]
                if aa_cdr3 in prop_dict and aa_epi in prop_dict:
                    # Interaction as absolute difference
                    val = abs(prop_dict[aa_cdr3] - prop_dict[aa_epi])
                    physchem_map[prop_idx, i, j] = val
    
    return physchem_map

# Collect unique CDR3β-epitope pairs from all splits
print("\n📊 Collecting unique CDR3β-epitope pairs...")
splits_dir = Path('splits')
pair_set = set()

for split_file in splits_dir.glob('*.tsv'):
    df = pd.read_csv(split_file, sep='\t')
    print(f"   {split_file.name:35s} : {len(df):6d} rows")
    
    for _, row in df.iterrows():
        if pd.notna(row['cdr3']) and pd.notna(row['antigen.epitope']):
            pair_key = f"{row['cdr3']}|{row['antigen.epitope']}"
            pair_set.add(pair_key)

pairs_list = sorted(list(pair_set))
print(f"\n📋 Total unique pairs: {len(pairs_list)}")

# Compute physicochemical maps
print("\n🧪 Computing physicochemical maps...")
physchem_maps = []
physchem_index = {}

for idx, pair_key in enumerate(tqdm(pairs_list, desc="Computing")):
    cdr3, epitope = pair_key.split('|')
    physchem_map = compute_physchem_map(cdr3, epitope)
    physchem_maps.append(physchem_map)
    physchem_index[pair_key] = idx

physchem_maps = np.array(physchem_maps, dtype=np.float32)
print(f"   Shape: {physchem_maps.shape}")

# Save
emb_dir = Path('embeddings')
emb_dir.mkdir(exist_ok=True)

np.save(emb_dir / 'physchem_maps.npy', physchem_maps)
with open(emb_dir / 'physchem_index.json', 'w') as f:
    json.dump(physchem_index, f)

physchem_size = (emb_dir / 'physchem_maps.npy').stat().st_size / 1e9

print(f"\n💾 Saved:")
print(f"   physchem_maps.npy   : {physchem_size:.1f} GB")
print(f"   physchem_index.json : {len(physchem_index)} pairs")
print("\n✅ Done!")
