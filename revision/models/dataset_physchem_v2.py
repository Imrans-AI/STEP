"""
Dataset for the normalised v2 splits (R2-1 / R2-4 / R2-7).

Differences from dataset_physchem_only.py:
  * reads physchem_index_v2.json, whose keys are already normalised
  * resolves map indices across TWO arrays:
        idx <  N_ORIG  -> physchem_maps.npy           (reused, never written)
        idx >= N_ORIG  -> physchem_maps_v2_extra.npy  (the repaired/new pairs)
  * caches the 7 global features per split to .npy, since `peptides` is slow and
    the training matrix reads the same splits 15 times
  * always returns lengths, so the length_only control can be run

Returns (physchem_map, global_features, label, cdr3_len, epi_len).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset_physchem_only import compute_global_features

MAX_CDR3, MAX_EPI = 20, 15
BASE = Path(__file__).parent.parent
EMB = BASE / 'data' / 'embeddings'
CACHE = BASE / 'data' / 'embeddings' / 'globalfeat_cache'


class PhysChemV2Dataset(Dataset):
    def __init__(self, data_file, verbose=True):
        self.data = pd.read_csv(data_file, sep='\t')
        self.name = Path(data_file).stem

        maps_main = np.load(EMB / 'physchem_maps.npy', mmap_mode='r')
        self.n_orig = maps_main.shape[0]
        self.maps_main = maps_main
        extra_path = EMB / 'physchem_maps_v2_extra.npy'
        self.maps_extra = np.load(extra_path, mmap_mode='r') if extra_path.exists() else None
        with open(EMB / 'physchem_index_v2.json') as f:
            self.index = json.load(f)

        keys = [f"{c}|{e}" for c, e in zip(self.data.cdr3, self.data['antigen.epitope'])]
        missing = [k for k in keys if k not in self.index]
        if missing:
            raise KeyError(f"{len(missing)} pairs absent from physchem_index_v2.json, "
                           f"e.g. {missing[:3]}")
        self.map_idx = np.array([self.index[k] for k in keys], dtype=np.int64)

        self.cdr3_len = np.array([min(len(c), MAX_CDR3) for c in self.data.cdr3], dtype=np.int64)
        self.epi_len = np.array([min(len(e), MAX_EPI) for e in self.data['antigen.epitope']],
                                dtype=np.int64)

        CACHE.mkdir(exist_ok=True)
        # namespace by the split DIRECTORY as well as the file stem: several split
        # directories use the same file names (train.tsv, val.tsv, test_balanced.tsv),
        # and a stem-only key makes them share a cache entry. The length check below
        # catches the mismatch whenever the row counts differ, but two different
        # splits of equal size would silently swap feature matrices.
        cache_file = CACHE / f"{Path(data_file).parent.name}__{self.name}_global.npy"
        if cache_file.exists():
            self.global_features = np.load(cache_file)
            if len(self.global_features) != len(self.data):
                cache_file.unlink()
            else:
                if verbose:
                    print(f"  global features: cached ({cache_file.name})")
        if not cache_file.exists():
            if verbose:
                print(f"  global features: computing for {len(self.data):,} rows…")
            gf = [compute_global_features(c, e)
                  for c, e in zip(self.data.cdr3, self.data['antigen.epitope'])]
            self.global_features = np.array(gf, dtype=np.float32)
            np.save(cache_file, self.global_features)
        self.labels = self.data['label'].to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        j = self.map_idx[i]
        arr = self.maps_main[j] if j < self.n_orig else self.maps_extra[j - self.n_orig]
        return (torch.from_numpy(np.asarray(arr)).float(),
                torch.from_numpy(self.global_features[i]).float(),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.cdr3_len[i], dtype=torch.long),
                torch.tensor(self.epi_len[i], dtype=torch.long))
