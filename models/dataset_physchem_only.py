"""
Dataset for physicochemical-only models (no ESM-2)
"""
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from pathlib import Path
import peptides

def compute_global_features(cdr3_seq, epitope_seq):
    """Compute DAISY's 7 global features"""
    hydrophobicity_table = peptides.tables.HYDROPHOBICITY["KyteDoolittle"]
    
    try:
        tcr_peptide = peptides.Peptide(cdr3_seq)
        epitope_peptide = peptides.Peptide(epitope_seq)
        
        tcr_features = [
            tcr_peptide.isoelectric_point(),
            tcr_peptide.instability_index(),
            tcr_peptide.aliphatic_index(),
            tcr_peptide.boman(),
            tcr_peptide.hydrophobic_moment(),
            tcr_peptide.molecular_weight(),
            tcr_peptide.auto_correlation(table=hydrophobicity_table, lag=1),
        ]
        
        epitope_features = [
            epitope_peptide.isoelectric_point(),
            epitope_peptide.instability_index(),
            epitope_peptide.aliphatic_index(),
            epitope_peptide.boman(),
            epitope_peptide.hydrophobic_moment(),
            epitope_peptide.molecular_weight(),
            epitope_peptide.auto_correlation(table=hydrophobicity_table, lag=1),
        ]
        
        global_diff = np.array([t - e for t, e in zip(tcr_features, epitope_features)])
    except:
        global_diff = np.zeros(7)
    
    return global_diff.astype(np.float32)

class PhysChemOnlyDataset(Dataset):
    """Physicochemical + Global features only (NO ESM-2!)"""
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, sep='\t')
        
        base_dir = Path(__file__).parent.parent
        emb_dir = base_dir / 'data' / 'embeddings'
        
        print(f"Loading physchem maps...")
        self.physchem_maps = np.load(emb_dir / 'physchem_maps.npy')
        with open(emb_dir / 'physchem_index.json', 'r') as f:
            self.physchem_index = json.load(f)
        print(f"✅ Loaded physchem maps")
        
        print(f"Computing global features for {len(self.data)} samples...")
        self.global_features = []
        for idx in range(len(self.data)):
            cdr3_seq = self.data.iloc[idx]['cdr3']
            epitope_seq = self.data.iloc[idx]['antigen.epitope']
            global_feat = compute_global_features(cdr3_seq, epitope_seq)
            self.global_features.append(global_feat)
        
        self.global_features = np.array(self.global_features)
        print(f"✅ Global features computed: shape {self.global_features.shape}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        pair_key = f"{row['cdr3']}|{row['antigen.epitope']}"
        physchem_idx = self.physchem_index[pair_key]
        physchem_map = torch.from_numpy(self.physchem_maps[physchem_idx]).float()
        
        global_feat = torch.from_numpy(self.global_features[idx]).float()
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return physchem_map, global_feat, label
