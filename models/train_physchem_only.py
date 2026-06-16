"""
Train Physicochemical-Only Model (No ESM-2)
This will be good for unseen epitopes
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from sklearn.metrics import roc_auc_score

from ensemble_model import PhysChemOnlyModel, compute_global_features
from dataset_hybrid import HybridTCREpitopeDataset

# Reproducibility
import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

# Hyperparameters
HIDDEN_DIM = 256
DROPOUT = 0.3
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
PATIENCE = 8
WEIGHT_DECAY = 1e-5

# Paths
BASE_DIR = Path(__file__).parent.parent
SPLITS = BASE_DIR / 'data' / 'splits'
CHECKPOINTS = Path(__file__).parent / 'checkpoints'

RUN_ID = f"physchem_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = CHECKPOINTS / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("  PHYSICOCHEMICAL-ONLY MODEL TRAINING")
print("  (For Ensemble - No ESM-2 embeddings)")
print("="*80)
print(f"Run ID: {RUN_ID}")
print("="*80)

def create_validation_set():
    print("\n📊 Creating validation set...")
    seen_df = pd.read_csv(f"{SPLITS}/seen_pair.tsv", sep='\t')
    train_df = pd.read_csv(f"{SPLITS}/train.tsv", sep='\t')
    train_neg = train_df[train_df['label'] == 0]
    
    sampled_neg = train_neg.sample(n=min(len(seen_df), len(train_neg)), random_state=42)
    val_df = pd.concat([seen_df, sampled_neg], ignore_index=True).sample(frac=1, random_state=42)
    
    val_path = f'/tmp/validation_physchem.tsv'
    val_df.to_csv(val_path, sep='\t', index=False)
    return val_path

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create datasets
    train_path = f"{SPLITS}/train.tsv"
    val_path = create_validation_set()
    
    print("\nLoading datasets...")
    train_dataset = HybridTCREpitopeDataset(train_path)
    val_dataset = HybridTCREpitopeDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"\nTraining samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Model
    model = PhysChemOnlyModel(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
    
    print("\n🚀 Starting training (physchem-only)...")
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_losses = []
        train_preds, train_labels = [], []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for cdr3, epi, phys, global_feat, labels in pbar:
            # Only use physchem + global (no ESM-2!)
            phys = phys.to(device)
            global_feat = global_feat.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(phys, global_feat)
            loss = criterion(outputs, labels.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': np.mean(train_losses[-100:])})
        
        # Validation
        model.eval()
        val_losses = []
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for cdr3, epi, phys, global_feat, labels in val_loader:
                phys = phys.to(device)
                global_feat = global_feat.to(device)
                labels = labels.to(device)
                
                outputs = model(phys, global_feat)
                loss = criterion(outputs, labels.float())
                
                val_losses.append(loss.item())
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        print(f"Epoch {epoch+1}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")
        
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'epoch': epoch
            }, RUN_DIR / 'best_model.pt')
            print(f"  ✅ New best model saved (Val AUC: {val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️  Early stopping after {epoch+1} epochs")
                break
    
    # Save history
    with open(RUN_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    with open(RUN_DIR / 'results.json', 'w') as f:
        json.dump({'best_val_auc': best_val_auc}, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {RUN_DIR}")

if __name__ == '__main__':
    train()
