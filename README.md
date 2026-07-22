# STEP: Structure-Informed TCR–Epitope Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

Official implementation of **STEP (Structure-Informed TCR–Epitope Prediction)**, a deep learning framework for predicting T-cell receptor (TCR) CDR3β–epitope binding.

> **STEP: Structure-Informed Physicochemical Representations Improve Generalization in TCR–Epitope Binding Prediction**
> Muhammad Imran1,2, [Syed Mansoor Jan1,2, Muhammad Rizwan1,2, Adil Farooq1,2, Waqar Ali1,2, Yanjing Wang3, Numan Yousaf1,2, Jiayi Li1,2, Wenji Ma5, Dongqing Wei*1,2,4,6]
> *Manuscript submitted to Cell Systems *, 2026

---

## Highlights

- **0.7549 ROC-AUC** on 581 unseen epitopes (balanced test set, n=3,314)
- **ROC-AUC 1.000** for the 5 clinical viral epitopes evaluable in the balanced test (CMV, EBV, HIV); **0.963–1.000** across all 6 on the full imbalanced test set
- **0.913 ROC-AUC** on melanoma cancer validation (n=1,582); **0.680** on the conservative overlap-free subset
- Outperforms 5 baselines (DAISY, ERGO-LSTM, ERGO-AE, TEINet, ATM-TCR) with p < 0.001
- Structure priors derived from **198 TCR–pMHC crystal structures** (STCRdab)

---

## Model Architecture

STEP integrates:
1. **5-channel physicochemical interaction maps** (5 × 20 × 15) encoding pairwise TCR–epitope residue properties (hydrophobicity, charge, flexibility, refractivity, area loss)
2. **Structure-informed weighting** using consensus contact probability maps from 198 PDB structures
3. **ResNet-style CNN** for local interaction feature extraction
4. **Dual Attention Network (DANet)** with position and channel attention branches
5. **7 global biophysical features** (Δ TCR–epitope)
6. **Adaptive average pooling + softmax** for binding probability prediction

---

## Installation

```bash
git clone https://github.com/Imrans-AI/STEP.git
cd STEP
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.x (built for your CUDA version; STEP was trained with CUDA 12.8)
- CUDA 12.x (recommended; CPU inference also supported)

> ANARCI (used for IMGT numbering of CDR3β) is most reliably installed via bioconda:
> `conda install -c bioconda anarci`

---

## Quick Start

### Download Pre-trained Checkpoint

Download from Zenodo: [DOI to be assigned upon publication]

```
models/checkpoints/step_600epi_20260402_164333/best_model.pt
```

### Run Inference

```python
import torch
from models.step_model import STEPModel
from models.dataset_physchem_only import PhysChemOnlyDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = STEPModel(use_structure_priors=True).to(device)
checkpoint = torch.load(
    'models/checkpoints/step_600epi_20260402_164333/best_model.pt',
    map_location=device, weights_only=False
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load data
dataset = PhysChemOnlyDataset('data/splits_600_epitope/test_balanced.tsv')
loader = DataLoader(dataset, batch_size=128, shuffle=False)

# Predict
predictions = []
with torch.no_grad():
    for physchem_map, global_features, labels in loader:
        physchem_map = physchem_map.to(device)
        global_features = global_features.to(device)
        outputs = model(physchem_map, global_features)
        probs = outputs[:, 1].cpu().numpy()
        predictions.extend(probs)
```

### Train from Scratch

```bash
cd models
python train_physchem_only.py
```

---

## Data

### Input Format

TSV file with columns:
| Column | Description |
|--------|-------------|
| `cdr3` | CDR3β amino acid sequence |
| `antigen.epitope` | Peptide epitope sequence (MHC Class I, 8–15 aa) |
| `label` | 1 = binding, 0 = non-binding |

### Compute PhysChem Maps

```bash
cd data
python compute_physchem.py
```

### Build Structure Priors

```bash
cd data
python structure_aware_physchem.py
# Raw PDB contacts:
python raw/extract_stcrdab_contacts.py
```

### Data Sources

| Database | Entries | Reference |
|----------|---------|-----------|
| VDJdb | 139,744 | Bagaev et al., 2020 |
| McPAS-TCR | 40,731 | Tickotsky et al., 2017 |
| IEDB | 2,742 | Vita et al., 2019 |
| STCRdab | 198 structures | Leem et al., 2018 |

After cross-database merging, filtering and deduplication: **99,323 unique positive pairs** spanning **1,181 epitopes** (93,654 unique CDR3β). The epitope-disjoint training split contains **131,974 samples** (97,666 positive + 34,308 negative) over **600 epitopes** and **92,718 unique CDR3β** (positive pairs).

> **Note:** Dataset TSVs and the trained checkpoint are hosted on Zenodo due to GitHub file-size limits.

---

## Reproducing Paper Results

### Main Results (Table 1)

```bash
python analysis/REDO_CLINICAL_ON_BALANCED.py
python analysis/PROPER_STATISTICAL_TESTS.py
```

### Cancer Validation

```bash
python analysis/UPDATE_CANCER_VALIDATION.py
```

### SHAP Interpretability

```bash
python analysis/RUN_SHAP_FIXED_PROPER.py
```

Channels are labelled in model order (hydrophobicity, charge, flexibility, refractivity, area loss), ranked by mean |SHAP|: hydrophobicity > refractivity > area loss > charge > flexibility.

### Figures

```bash
python figures/CREATE_FINAL_8SUBPLOT_FIGURE.py  # Figure 2: baseline benchmarking
python figures/FIGURES_NMI_V3.py                # Figures 3 & 4: per-epitope + cancer validation
python figures/BUILD_FIG5_DAISY_PATTERN.py      # Figure 5: SHAP + ablation
python figures/FIG6_FINAL_V6.py                 # Figure 6: Score-CAM
```

---

## Repository Structure

```
STEP/
├── models/
│   ├── step_model.py              # STEP model architecture
│   ├── dataset_physchem_only.py   # Dataset class
│   └── train_physchem_only.py     # Training script
├── data/
│   ├── compute_physchem.py        # Compute 5-channel physchem maps
│   ├── structure_aware_physchem.py # Build structure prior contact maps
│   └── raw/
│       └── extract_stcrdab_contacts.py  # PDB contact extraction
├── analysis/
│   ├── REDO_CLINICAL_ON_BALANCED.py
│   ├── RUN_SHAP_FIXED_PROPER.py
│   ├── UPDATE_CANCER_VALIDATION.py
│   └── PROPER_STATISTICAL_TESTS.py
├── figures/
│   ├── CREATE_FINAL_8SUBPLOT_FIGURE.py  # Figure 2 (baseline benchmarking)
│   ├── FIGURES_NMI_V3.py                # Figures 3 & 4 (per-epitope + cancer)
│   ├── BUILD_FIG5_DAISY_PATTERN.py      # Figure 5 (SHAP + ablation)
│   └── FIG6_FINAL_V6.py                 # Figure 6 (Score-CAM)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Performance

### Balanced Test Set (n = 3,314, 581 unseen epitopes)

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| **STEP** | **0.7549** | **0.7356** |
| DAISY | 0.7013 | 0.6582 |
| ERGO-LSTM | 0.5889 | 0.5834 |
| ERGO-AE | 0.5695 | 0.5644 |
| TEINet | 0.4933 | 0.5178 |
| ATM-TCR | 0.4790 | 0.4858 |

All differences vs STEP: p < 0.001 (DeLong's test).

### Cancer Validation

| Dataset | ROC-AUC (full) | 95% CI | ROC-AUC (overlap-free)¹ | 95% CI | n (full / overlap-free) |
|---------|----------------|--------|-------------------------|--------|-------------------------|
| Melanoma | 0.913 | [0.898–0.926] | 0.680 | [0.584–0.765] | 1,582 / 132 |
| Lung | 0.719 | [0.644–0.803] | 0.615 | [0.498–0.728] | 170 / 96 |

MART-1/Melan-A melanoma antigen: ROC-AUC **0.983**.

¹ Overlap-free = after removing TCR–epitope pairs shared with the training set; the conservative estimate of novel-antigen generalization. The lung overlap-free CI crosses 0.5 and is not significantly above chance.

---

## Data and Model Availability

Processed datasets and the trained model checkpoint are archived on Zenodo:
**DOI: [10.5281/zenodo.21473727](https://doi.org/10.5281/zenodo.21473727)** (CC-BY-4.0)

Contents: 600-epitope train/validation/test splits; cancer-specific held-out test sets (melanoma, lung) in full and overlap-free forms; consensus structure-prior contact map; physicochemical index; trained STEP checkpoint (`best_model.pt`).

Physicochemical interaction maps (`physchem_maps.npy`) are regenerable from the provided splits via `compute_physchem.py`. Raw source databases (VDJdb, McPAS-TCR, IEDB, STCRdab) are available from their respective public repositories.

---
## Citation

If you use STEP in your research, please cite:

```bibtex
@article{imran2026step,
  title={STEP: Structure-Informed Physicochemical Representations Improve Generalization in TCR--Epitope Binding Prediction},
  author={Imran, Muhammad and Jan, Syed Mansoor and Rizwan, Muhammad and Farooq, Adil and Ali, Waqar and Wang, Yanjing and Yousaf, Numan and Li, Jiayi and Ma, Wenji and Liu, Junhe and Wei, Dongqing},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  note={Manuscript under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

Muhammad Imran — SJTU
For questions, open an issue or contact [imran93@sjtu.edu.cn].
