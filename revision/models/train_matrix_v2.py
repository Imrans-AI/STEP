"""
Training matrix on the normalised v2 splits — R1-2, R2-4, R2-7.

Five prior variants x three seeds. Hyperparameters are copied from
train_step_600epi.py and unchanged: batch 256, lr 1e-4, AdamW, weight decay 1e-5,
BCELoss, ReduceLROnPlateau(mode='max', factor=0.5, patience=4), grad-norm clip 1.0,
40 epochs, early-stopping patience 10.

Two deliberate differences from train_step_600epi.py:
  * seeds are set explicitly (that script set none at all), so R2-7 can be answered
  * the dataset is the physchem-only v2 loader, not HybridTCREpitopeDataset, which
    also loads 9.6 GB of unused ESM-2 embeddings. The model consumes only
    physchem_map and global_features, so this does not change what is trained.

Checkpoint selection is on val.tsv ONLY. test_balanced.tsv is never read here.

Usage:  python train_matrix_v2.py --prior corrected --seed 42
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from step_model import STEPModel
from dataset_physchem_v2 import PhysChemV2Dataset
from prior_variants import get_prior, VARIANTS

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 40
PATIENCE = 10
WEIGHT_DECAY = 1e-5

BASE = Path(__file__).parent.parent
SPLITS = BASE / 'data' / 'splits_600_epitope_v2'
CKPT_ROOT = Path(__file__).parent / 'checkpoints_v2'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior', required=True, choices=VARIANTS)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--workers', type=int, default=4)
    # R2-1: the same training recipe has to run on the similarity-disjoint split too.
    # Overriding the paths keeps every hyperparameter identical by construction,
    # rather than copying this script and letting the two drift.
    ap.add_argument('--splits', default=str(SPLITS))
    ap.add_argument('--ckpt-root', default=str(CKPT_ROOT))
    args = ap.parse_args()
    splits = Path(args.splits)
    ckpt_root = Path(args.ckpt_root)

    run = f"{args.prior}_seed{args.seed}"
    out = ckpt_root / run
    out.mkdir(parents=True, exist_ok=True)
    done = out / 'results.json'
    if done.exists():
        print(f"[{run}] already complete — skipping")
        return

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{run}] device={device} splits={splits.name} ckpt_root={ckpt_root.name}")

    W, use_mask = get_prior(args.prior, args.seed)
    print(f"[{run}] prior mean={W.mean():.4f} std={W.std():.4f} length_mask={use_mask}")

    train_ds = PhysChemV2Dataset(splits / 'train.tsv')
    val_ds = PhysChemV2Dataset(splits / 'val.tsv')
    print(f"[{run}] train={len(train_ds):,} val={len(val_ds):,}")

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=args.workers, pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # build the model, then overwrite the prior buffer with this variant
    model = STEPModel(use_structure_priors=True, use_length_mask=use_mask,
                      priors_file='structure_priors_v2.npz').to(device)
    with torch.no_grad():
        model.contact_weights.copy_(torch.from_numpy(W).unsqueeze(0).to(device))
    assert np.allclose(model.contact_weights.squeeze(0).cpu().numpy(), W), "prior not installed"

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    best_val_auc, patience_counter = 0.0, 0
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        losses, preds, labs = [], [], []
        for phys, gf, y, cl, el in train_loader:
            phys, gf, y = phys.to(device), gf.to(device), y.to(device)
            cl, el = cl.to(device), el.to(device)
            optimizer.zero_grad()
            o = model(phys, gf, cdr3_len=cl, epi_len=el) if use_mask else model(phys, gf)
            loss = criterion(o, y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            preds.extend(o.detach().cpu().numpy()); labs.extend(y.cpu().numpy())
        tr_loss, tr_auc = float(np.mean(losses)), roc_auc_score(labs, preds)

        model.eval()
        vlosses, vpreds, vlabs = [], [], []
        with torch.no_grad():
            for phys, gf, y, cl, el in val_loader:
                phys, gf, y = phys.to(device), gf.to(device), y.to(device)
                cl, el = cl.to(device), el.to(device)
                o = model(phys, gf, cdr3_len=cl, epi_len=el) if use_mask else model(phys, gf)
                vlosses.append(criterion(o, y.float()).item())
                vpreds.extend(o.cpu().numpy()); vlabs.extend(y.cpu().numpy())
        va_loss, va_auc = float(np.mean(vlosses)), roc_auc_score(vlabs, vpreds)

        history['train_loss'].append(tr_loss); history['train_auc'].append(tr_auc)
        history['val_loss'].append(va_loss); history['val_auc'].append(va_auc)
        print(f"[{run}] epoch {epoch+1:>2d}  train_auc={tr_auc:.4f}  val_auc={va_auc:.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

        scheduler.step(va_auc)
        if va_auc > best_val_auc:
            best_val_auc = va_auc
            torch.save({'model_state_dict': model.state_dict(), 'val_auc': va_auc,
                        'epoch': epoch, 'prior': args.prior, 'seed': args.seed},
                       out / 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[{run}] early stop at epoch {epoch+1}")
                break

    json.dump(history, open(out / 'history.json', 'w'), indent=2)
    json.dump({'best_val_auc': best_val_auc, 'prior': args.prior, 'seed': args.seed,
               'epochs_run': len(history['val_auc']), 'minutes': (time.time() - t0) / 60},
              open(done, 'w'), indent=2)
    print(f"[{run}] DONE best_val_auc={best_val_auc:.4f} "
          f"in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
