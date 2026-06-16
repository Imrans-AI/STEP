"""
Comprehensive Cancer Validation - Update existing analysis
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from models.step_model import STEPModel
from models.dataset_physchem_only import PhysChemOnlyDataset
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("  COMPREHENSIVE CANCER VALIDATION")
print("="*80)

# ============================================================================
# Step 1: Examine Cancer Data
# ============================================================================
print("\n1️⃣  EXAMINING CANCER DATA")
print("="*80)

# Load all cancer data
melanoma_all = pd.read_csv('data/cancer_specific/melanoma_all_sources.tsv', sep='\t')
melanoma_test = pd.read_csv('data/cancer_specific/melanoma_test.tsv', sep='\t')
lung_all = pd.read_csv('data/cancer_specific/lung_all_sources.tsv', sep='\t')
lung_test = pd.read_csv('data/cancer_specific/lung_test.tsv', sep='\t')

print(f"\n📊 MELANOMA Data:")
print(f"   All sources: {len(melanoma_all)} samples")
print(f"   Test set:    {len(melanoma_test)} samples")
print(f"   Columns: {list(melanoma_all.columns)}")
if 'label' in melanoma_test.columns:
    print(f"   Test positive: {melanoma_test['label'].sum()} ({melanoma_test['label'].mean():.1%})")
print(f"\n   First 5 samples:")
print(melanoma_test.head().to_string())

print(f"\n📊 LUNG CANCER Data:")
print(f"   All sources: {len(lung_all)} samples")
print(f"   Test set:    {len(lung_test)} samples")
print(f"   Columns: {list(lung_all.columns)}")
if 'label' in lung_test.columns:
    print(f"   Test positive: {lung_test['label'].sum()} ({lung_test['label'].mean():.1%})")
print(f"\n   First 5 samples:")
print(lung_test.head().to_string())

# ============================================================================
# Step 2: Check Epitopes
# ============================================================================
print("\n" + "="*80)
print("2️⃣  EPITOPE ANALYSIS")
print("="*80)

print(f"\n🎯 Melanoma Epitopes:")
mel_epi = melanoma_test['antigen.epitope'].value_counts() if 'antigen.epitope' in melanoma_test.columns else melanoma_all['antigen.epitope'].value_counts()
print(mel_epi.head(10).to_string())

print(f"\n🎯 Lung Cancer Epitopes:")
lung_epi = lung_test['antigen.epitope'].value_counts() if 'antigen.epitope' in lung_test.columns else lung_all['antigen.epitope'].value_counts()
print(lung_epi.head(10).to_string())

# Known cancer antigens
cancer_antigens = {
    # Melanoma antigens
    'TPRVTGGGAM': {'Cancer': 'Melanoma', 'Antigen': 'NY-ESO-1', 'HLA': 'A*02:01'},
    'EAAGIGILTV': {'Cancer': 'Melanoma', 'Antigen': 'MART-1', 'HLA': 'A*02:01'},
    'ELAGIGILTV': {'Cancer': 'Melanoma', 'Antigen': 'MART-1', 'HLA': 'A*02:01'},
    'AAGIGILTV':  {'Cancer': 'Melanoma', 'Antigen': 'MART-1', 'HLA': 'A*02:01'},
    'YMDGTMSQV':  {'Cancer': 'Melanoma', 'Antigen': 'Tyrosinase', 'HLA': 'A*02:01'},
    'IMDQVPFSV':  {'Cancer': 'Melanoma', 'Antigen': 'gp100', 'HLA': 'A*02:01'},
    'ITDQVPFSV':  {'Cancer': 'Melanoma', 'Antigen': 'gp100', 'HLA': 'A*02:01'},
    'KTWGQYWQV':  {'Cancer': 'Melanoma', 'Antigen': 'gp100', 'HLA': 'A*02:01'},
    'SLLMWITQV':  {'Cancer': 'Melanoma', 'Antigen': 'NY-ESO-1', 'HLA': 'A*02:01'},
    'SLLMWITQC':  {'Cancer': 'Melanoma', 'Antigen': 'NY-ESO-1', 'HLA': 'A*02:01'},
    
    # Lung cancer antigens
    'KLVVVGADGV': {'Cancer': 'Lung', 'Antigen': 'KRAS G12V', 'HLA': 'A*11:01'},
    'YLDLALMSV':  {'Cancer': 'Lung', 'Antigen': 'p53', 'HLA': 'A*02:01'},
    'KSFKVDQNL':  {'Cancer': 'Lung', 'Antigen': 'NY-ESO-1', 'HLA': 'A*02:01'},
}

# Find overlap
all_test_epitopes = set(mel_epi.index.tolist() + lung_epi.index.tolist())
known_in_data = {}
for epi in all_test_epitopes:
    if epi in cancer_antigens:
        known_in_data[epi] = cancer_antigens[epi]

print(f"\n📚 Known Cancer Antigens in Test Data:")
print(f"   Total found: {len(known_in_data)}")
for epi, info in known_in_data.items():
    print(f"   {epi}: {info['Antigen']} ({info['Cancer']}, {info['HLA']})")

# ============================================================================
# Step 3: Run STEP on Cancer Data
# ============================================================================
print("\n" + "="*80)
print("3️⃣  RUNNING STEP ON CANCER DATA")
print("="*80)

device = torch.device('cuda')

# Load model
print("\n📥 Loading STEP model...")
model = STEPModel(use_structure_priors=True).to(device)
checkpoint = torch.load('models/checkpoints/step_600epi_20260402_164333/best_model.pt', 
                       map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def evaluate_dataset(test_file, name):
    """Evaluate STEP on a dataset"""
    print(f"\n🧬 Testing on {name}...")
    
    try:
        dataset = PhysChemOnlyDataset(test_file)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
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
        
        # Compute metrics if we have both classes
        if len(np.unique(all_labels)) > 1:
            roc = roc_auc_score(all_labels, all_preds)
            pr = average_precision_score(all_labels, all_preds)
            print(f"   ✓ {name}:")
            print(f"     Samples: {len(all_labels)}")
            print(f"     Positive: {int(all_labels.sum())} ({all_labels.mean():.1%})")
            print(f"     ROC-AUC: {roc:.4f}")
            print(f"     PR-AUC:  {pr:.4f}")
            return {
                'name': name,
                'samples': len(all_labels),
                'positive': int(all_labels.sum()),
                'roc_auc': roc,
                'pr_auc': pr,
                'predictions': all_preds,
                'labels': all_labels
            }
        else:
            print(f"   ⚠️  Only one class in {name}")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

# Evaluate both datasets
mel_results = evaluate_dataset('data/cancer_specific/melanoma_test.tsv', 'Melanoma')
lung_results = evaluate_dataset('data/cancer_specific/lung_test.tsv', 'Lung Cancer')

# ============================================================================
# Step 4: Per-Epitope Cancer Performance
# ============================================================================
print("\n" + "="*80)
print("4️⃣  PER-EPITOPE CANCER PERFORMANCE")
print("="*80)

def per_epitope_analysis(test_file, results, name):
    """Analyze performance per cancer epitope"""
    if results is None:
        return None
    
    df = pd.read_csv(test_file, sep='\t')
    df['prediction'] = results['predictions']
    
    epi_results = []
    for epitope in df['antigen.epitope'].unique():
        epi_data = df[df['antigen.epitope'] == epitope]
        n_pos = epi_data['label'].sum()
        n_neg = len(epi_data) - n_pos
        
        if n_pos >= 2 and n_neg >= 2:
            try:
                roc = roc_auc_score(epi_data['label'], epi_data['prediction'])
                pr = average_precision_score(epi_data['label'], epi_data['prediction'])
                
                # Check if known antigen
                antigen_info = cancer_antigens.get(epitope, {})
                
                epi_results.append({
                    'Epitope': epitope,
                    'Cancer': antigen_info.get('Cancer', 'Unknown'),
                    'Antigen': antigen_info.get('Antigen', 'Unknown'),
                    'HLA': antigen_info.get('HLA', 'Unknown'),
                    'N_Samples': len(epi_data),
                    'N_Pos': int(n_pos),
                    'ROC_AUC': roc,
                    'PR_AUC': pr,
                    'Mean_Pred': epi_data['prediction'].mean()
                })
            except:
                pass
    
    if epi_results:
        df_results = pd.DataFrame(epi_results).sort_values('ROC_AUC', ascending=False)
        print(f"\n📊 {name} - Per Epitope Performance:")
        print(df_results.head(10).to_string(index=False))
        return df_results
    return None

mel_per_epi = per_epitope_analysis('data/cancer_specific/melanoma_test.tsv', mel_results, 'Melanoma')
lung_per_epi = per_epitope_analysis('data/cancer_specific/lung_test.tsv', lung_results, 'Lung Cancer')

# Save results
if mel_per_epi is not None:
    mel_per_epi.to_csv('results/cancer_melanoma_per_epitope.csv', index=False)
    print(f"\n💾 Saved: results/cancer_melanoma_per_epitope.csv")

if lung_per_epi is not None:
    lung_per_epi.to_csv('results/cancer_lung_per_epitope.csv', index=False)
    print(f"💾 Saved: results/cancer_lung_per_epitope.csv")

# ============================================================================
# Step 5: Comprehensive Cancer Table
# ============================================================================
print("\n" + "="*80)
print("5️⃣  COMPREHENSIVE CANCER VALIDATION TABLE")
print("="*80)

cancer_summary = []

if mel_results:
    cancer_summary.append({
        'Cancer_Type': 'Melanoma',
        'Total_Samples': mel_results['samples'],
        'Positive_Pairs': mel_results['positive'],
        'Negative_Pairs': mel_results['samples'] - mel_results['positive'],
        'Pos_Rate': f"{mel_results['positive']/mel_results['samples']*100:.1f}%",
        'STEP_ROC_AUC': f"{mel_results['roc_auc']:.4f}",
        'STEP_PR_AUC': f"{mel_results['pr_auc']:.4f}",
        'Known_Antigens_Tested': sum(1 for e in mel_per_epi['Epitope'] if e in cancer_antigens) if mel_per_epi is not None else 0,
        'Best_Epitope': mel_per_epi.iloc[0]['Epitope'] if mel_per_epi is not None and len(mel_per_epi) > 0 else 'N/A',
        'Best_Epitope_AUC': f"{mel_per_epi.iloc[0]['ROC_AUC']:.4f}" if mel_per_epi is not None and len(mel_per_epi) > 0 else 'N/A'
    })

if lung_results:
    cancer_summary.append({
        'Cancer_Type': 'Lung',
        'Total_Samples': lung_results['samples'],
        'Positive_Pairs': lung_results['positive'],
        'Negative_Pairs': lung_results['samples'] - lung_results['positive'],
        'Pos_Rate': f"{lung_results['positive']/lung_results['samples']*100:.1f}%",
        'STEP_ROC_AUC': f"{lung_results['roc_auc']:.4f}",
        'STEP_PR_AUC': f"{lung_results['pr_auc']:.4f}",
        'Known_Antigens_Tested': sum(1 for e in lung_per_epi['Epitope'] if e in cancer_antigens) if lung_per_epi is not None else 0,
        'Best_Epitope': lung_per_epi.iloc[0]['Epitope'] if lung_per_epi is not None and len(lung_per_epi) > 0 else 'N/A',
        'Best_Epitope_AUC': f"{lung_per_epi.iloc[0]['ROC_AUC']:.4f}" if lung_per_epi is not None and len(lung_per_epi) > 0 else 'N/A'
    })

cancer_summary_df = pd.DataFrame(cancer_summary)
print("\n" + cancer_summary_df.to_string(index=False))

cancer_summary_df.to_csv('results/cancer_validation_summary.csv', index=False)
print(f"\n💾 Saved: results/cancer_validation_summary.csv")

# ============================================================================
# Step 6: Visualization
# ============================================================================
print("\n" + "="*80)
print("6️⃣  CREATING CANCER VALIDATION FIGURES")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Cancer overall performance
ax1 = axes[0, 0]
if mel_results and lung_results:
    cancers = ['Melanoma', 'Lung']
    roc_vals = [mel_results['roc_auc'], lung_results['roc_auc']]
    pr_vals = [mel_results['pr_auc'], lung_results['pr_auc']]
    
    x = np.arange(len(cancers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, roc_vals, width, label='ROC-AUC', color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x + width/2, pr_vals, width, label='PR-AUC', color='#e74c3c', edgecolor='black')
    
    for bars, vals in [(bars1, roc_vals), (bars2, pr_vals)]:
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(cancers, fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax1.set_title('STEP Performance on Cancer Datasets', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

# Plot 2: Melanoma per-epitope
ax2 = axes[0, 1]
if mel_per_epi is not None and len(mel_per_epi) > 0:
    top_mel = mel_per_epi.head(10)
    colors = ['green' if c == 'Melanoma' else 'gray' for c in top_mel['Cancer']]
    bars = ax2.barh(range(len(top_mel)), top_mel['ROC_AUC'], color=colors, edgecolor='black')
    ax2.set_yticks(range(len(top_mel)))
    ax2.set_yticklabels(top_mel['Epitope'], fontsize=10)
    ax2.set_xlabel('ROC-AUC', fontsize=12, fontweight='bold')
    ax2.set_title('Melanoma: Top 10 Epitopes', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1.05])
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

# Plot 3: Lung per-epitope
ax3 = axes[1, 0]
if lung_per_epi is not None and len(lung_per_epi) > 0:
    top_lung = lung_per_epi.head(10)
    colors = ['green' if c == 'Lung' else 'gray' for c in top_lung['Cancer']]
    bars = ax3.barh(range(len(top_lung)), top_lung['ROC_AUC'], color=colors, edgecolor='black')
    ax3.set_yticks(range(len(top_lung)))
    ax3.set_yticklabels(top_lung['Epitope'], fontsize=10)
    ax3.set_xlabel('ROC-AUC', fontsize=12, fontweight='bold')
    ax3.set_title('Lung Cancer: Top 10 Epitopes', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, 1.05])
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

# Plot 4: Score distributions
ax4 = axes[1, 1]
if mel_results and lung_results:
    ax4.hist(mel_results['predictions'], bins=30, alpha=0.5, label='Melanoma', color='red', density=True)
    ax4.hist(lung_results['predictions'], bins=30, alpha=0.5, label='Lung Cancer', color='blue', density=True)
    ax4.set_xlabel('Prediction Score', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax4.set_title('Score Distribution: Cancer Datasets', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(alpha=0.3)

plt.suptitle('STEP Cancer Validation Analysis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('results/cancer_validation_comprehensive.png', dpi=300, bbox_inches='tight')
plt.savefig('results/cancer_validation_comprehensive.pdf', bbox_inches='tight')
plt.close()
print("✓ Saved: results/cancer_validation_comprehensive.png/pdf")

print("\n" + "="*80)
print("✅ CANCER VALIDATION COMPLETE!")
print("="*80)

print(f"""
SUMMARY:
  ✓ Melanoma: ROC={mel_results['roc_auc']:.4f}, PR={mel_results['pr_auc']:.4f}
  ✓ Lung:     ROC={lung_results['roc_auc']:.4f}, PR={lung_results['pr_auc']:.4f}

Files Created:
  📄 results/cancer_validation_summary.csv
  📄 results/cancer_melanoma_per_epitope.csv
  📄 results/cancer_lung_per_epitope.csv
  📄 results/cancer_validation_comprehensive.png/pdf
""")

