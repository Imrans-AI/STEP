"""
Re-run per-epitope and clinical relevance analysis on BALANCED data
For consistency with cancer validation and baseline comparison
"""
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from models.step_model import STEPModel
from models.dataset_physchem_only import PhysChemOnlyDataset
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("  PER-EPITOPE ANALYSIS ON BALANCED TEST SET")
print("="*80)

device = torch.device('cuda')

# Load model
print("\n📥 Loading STEP model...")
model = STEPModel(use_structure_priors=True).to(device)
checkpoint = torch.load('models/checkpoints/step_600epi_20260402_164333/best_model.pt', 
                       map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use BALANCED test set
print("\n📥 Loading BALANCED test set...")
test_file = 'data/splits_600_epitope/test_balanced.tsv'
dataset = PhysChemOnlyDataset(test_file)
loader = DataLoader(dataset, batch_size=128, shuffle=False)
print(f"   Loaded {len(dataset)} samples")

# Generate predictions
print("\n🔮 Generating predictions...")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in loader:
        physchem, global_feat, labels = batch
        outputs = model(physchem.to(device), global_feat.to(device))
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Verify
overall_roc = roc_auc_score(all_labels, all_preds)
overall_pr = average_precision_score(all_labels, all_preds)
print(f"\n📊 Overall on Balanced:")
print(f"   ROC-AUC: {overall_roc:.4f}")
print(f"   PR-AUC:  {overall_pr:.4f}")

# Save
test_df = pd.read_csv(test_file, sep='\t')
results_df = test_df.copy()
results_df['prediction'] = all_preds[:len(test_df)]
results_df.to_csv('results/STEP_predictions_balanced.csv', index=False)
print(f"💾 Saved: results/STEP_predictions_balanced.csv")

# Per-Epitope Analysis on Balanced
print("\n" + "="*80)
print("  PER-EPITOPE PERFORMANCE (BALANCED)")
print("="*80)

epitopes = results_df['antigen.epitope'].unique()
print(f"\nAnalyzing {len(epitopes)} unique epitopes...")

per_epitope_results = []
for epitope in epitopes:
    epi_data = results_df[results_df['antigen.epitope'] == epitope]
    n_pos = epi_data['label'].sum()
    n_neg = len(epi_data) - n_pos
    
    if n_pos >= 2 and n_neg >= 2:
        try:
            roc_auc = roc_auc_score(epi_data['label'], epi_data['prediction'])
            pr_auc = average_precision_score(epi_data['label'], epi_data['prediction'])
            
            per_epitope_results.append({
                'Epitope': epitope,
                'Length': len(epitope),
                'N_Samples': len(epi_data),
                'N_Pos': int(n_pos),
                'N_Neg': int(n_neg),
                'Pos_Rate': n_pos / len(epi_data),
                'ROC_AUC': roc_auc,
                'PR_AUC': pr_auc
            })
        except:
            pass

df_per_epi = pd.DataFrame(per_epitope_results)
df_per_epi = df_per_epi.sort_values('ROC_AUC', ascending=False)

print(f"\n✅ Computed metrics for {len(df_per_epi)} epitopes (≥2 pos & ≥2 neg)")

# Add clinical annotation
clinical_epitopes = {
    'NLVPMVATV': {'Virus': 'CMV', 'Protein': 'pp65'},
    'NIVPMVATV': {'Virus': 'CMV', 'Protein': 'pp65 variant'},
    'NMVPMVATV': {'Virus': 'CMV', 'Protein': 'pp65 variant'},
    'LLVPMVATV': {'Virus': 'CMV', 'Protein': 'pp65 variant'},
    'VLVPMVATV': {'Virus': 'CMV', 'Protein': 'pp65 variant'},
    'NLLPMVATV': {'Virus': 'CMV', 'Protein': 'pp65 variant'},
    'NCVPMVATV': {'Virus': 'CMV', 'Protein': 'pp65 variant'},
    'QASGNHAAGILTM': {'Virus': 'EBV', 'Protein': 'EBNA3C'},
    'FLKEQGGL': {'Virus': 'HIV', 'Protein': 'gag'},
}

df_per_epi['Virus'] = df_per_epi['Epitope'].apply(
    lambda x: clinical_epitopes[x]['Virus'] if x in clinical_epitopes else 'Unknown'
)
df_per_epi['Protein'] = df_per_epi['Epitope'].apply(
    lambda x: clinical_epitopes[x]['Protein'] if x in clinical_epitopes else 'Unknown'
)

df_per_epi.to_csv('results/per_epitope_performance_balanced.csv', index=False)
print(f"💾 Saved: results/per_epitope_performance_balanced.csv")

# Top 20 on balanced
print("\n" + "="*80)
print("  TOP 20 EPITOPES (BALANCED)")
print("="*80)
top_20 = df_per_epi.head(20)
print("\n" + top_20[['Epitope', 'N_Samples', 'N_Pos', 'ROC_AUC', 'PR_AUC', 'Virus']].to_string(index=False))

# Clinical epitopes
print("\n" + "="*80)
print("  CLINICAL EPITOPES PERFORMANCE (BALANCED)")
print("="*80)

clinical_subset = df_per_epi[df_per_epi['Virus'] != 'Unknown'].sort_values('ROC_AUC', ascending=False)
print(f"\nFound {len(clinical_subset)} clinical epitopes:")
print(clinical_subset[['Epitope', 'Virus', 'Protein', 'N_Samples', 'ROC_AUC', 'PR_AUC']].to_string(index=False))

if len(clinical_subset) > 0:
    print(f"\n📊 Clinical epitopes mean ROC-AUC (BALANCED): {clinical_subset['ROC_AUC'].mean():.4f}")
    print(f"📊 All epitopes mean ROC-AUC (BALANCED):      {df_per_epi['ROC_AUC'].mean():.4f}")

# Performance distribution
print("\n" + "="*80)
print("  PERFORMANCE DISTRIBUTION (BALANCED)")
print("="*80)

print(f"""
Overall Test Performance (BALANCED):
  ROC-AUC: {overall_roc:.4f}
  PR-AUC:  {overall_pr:.4f}

Per-Epitope Statistics ({len(df_per_epi)} epitopes):
  Mean ROC-AUC: {df_per_epi['ROC_AUC'].mean():.4f} ± {df_per_epi['ROC_AUC'].std():.4f}
  Median:       {df_per_epi['ROC_AUC'].median():.4f}
  Min:          {df_per_epi['ROC_AUC'].min():.4f}
  Max:          {df_per_epi['ROC_AUC'].max():.4f}

Distribution:
  ROC-AUC ≥ 0.9:  {(df_per_epi['ROC_AUC'] >= 0.9).sum()} epitopes ({(df_per_epi['ROC_AUC'] >= 0.9).sum()/len(df_per_epi)*100:.1f}%)
  ROC-AUC ≥ 0.8:  {(df_per_epi['ROC_AUC'] >= 0.8).sum()} epitopes ({(df_per_epi['ROC_AUC'] >= 0.8).sum()/len(df_per_epi)*100:.1f}%)
  ROC-AUC ≥ 0.7:  {(df_per_epi['ROC_AUC'] >= 0.7).sum()} epitopes ({(df_per_epi['ROC_AUC'] >= 0.7).sum()/len(df_per_epi)*100:.1f}%)
""")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE - ALL ON BALANCED DATA NOW!")
print("="*80)

