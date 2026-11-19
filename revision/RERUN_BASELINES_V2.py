#!/usr/bin/env python
"""
R2-3 baseline rerun on data/splits_600_epitope_v2/test_balanced.tsv.

Every model is scored on the SAME dataframe in the SAME row order, so the saved
per-sample predictions are directly pairable (this is what R1-5's paired DeLong needs).

Corrections applied relative to the submitted evaluation (see
revision/evidence/R2-3_baseline_audit.md):
  * TEINet   - the authors' real released weights (teinet_data.pth) loaded with the
               authors' own code, strict=True. The submitted number came from a
               randomly initialised STEPModel (0 of 163 keys loaded, strict=False).
  * ATM-TCR  - the repo's own encoding: AMINO_MAP indices, pad token 24, pad_type='mid'.
               The submitted number end-padded with index 20, which is amino acid 'B'.
  * ERGO-AE  - canonical TCR one-hot index (A=0..V=19, X=20) via the authors' own
               evaluation_methods.predict. The submitted script shifted every index by +1.
  * ERGO-LSTM, DAISY - unchanged, rerun to confirm they reproduce.

Every load is strict=True and asserted. Usage:
    python RERUN_BASELINES_V2.py
"""
import os, sys, json, random, tempfile, argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'models'))   # models/* import each other by bare name
os.chdir(ROOT)

# Default is the v2 split. Passing the v1 test file instead isolates how much of the
# old-vs-new change is the loading/encoding fix and how much is the change of split;
# in that mode STEP is skipped (no v1-trained 'corrected' checkpoint in checkpoints_v2).
TEST = sys.argv[1] if len(sys.argv) > 1 else 'data/splits_600_epitope_v2/test_balanced.tsv'
V2 = 'v2' in TEST
OUTDIR = 'results/baseline_rerun_v2' if V2 else 'results/baseline_rerun_v1check'
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); os.environ['PYTHONHASHSEED'] = str(s)


def strict_load(model, state, name):
    """Load strict=True, report expected vs loaded, assert nothing was silently skipped."""
    msd = model.state_dict()
    mk, ck = set(msd), set(state)
    shape_bad = [k for k in mk & ck if msd[k].shape != state[k].shape]
    info = dict(expects=len(mk), ckpt_has=len(ck), missing=sorted(mk - ck),
                unexpected=sorted(ck - mk), shape_mismatch=shape_bad)
    print(f"    keys: model expects {len(mk)}, ckpt has {len(ck)}, "
          f"missing {len(mk - ck)}, unexpected {len(ck - mk)}, shape-mismatch {len(shape_bad)}")
    if info['missing'] or info['unexpected'] or shape_bad:
        print(f"    !! MISMATCH for {name} -- STOPPING, not forcing the load")
        print(f"       missing   : {info['missing'][:10]}")
        print(f"       unexpected: {info['unexpected'][:10]}")
        raise SystemExit(f"{name}: key mismatch; refusing to load non-strictly.")
    model.load_state_dict(state)           # strict=True (default)
    info['loaded'] = len(mk)
    print(f"    strict=True load OK: {len(mk)}/{len(mk)} loaded")
    return info


def describe(name, p, y):
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    d = dict(n=int(p.size), min=float(p.min()), max=float(p.max()),
             mean=float(p.mean()), sd=float(p.std(ddof=1)),
             distinct=int(np.unique(p).size),
             distinct_6dp=int(np.unique(np.round(p, 6)).size),
             roc=float(roc_auc_score(y, p)), pr=float(average_precision_score(y, p)))
    flag = '  <-- NEAR-CONSTANT, MODEL NOT RUNNING' if d['sd'] < 0.01 or d['distinct_6dp'] < 100 else ''
    print(f"    preds: min {d['min']:.4g}  max {d['max']:.4g}  mean {d['mean']:.4f}  "
          f"SD {d['sd']:.4f}  distinct {d['distinct']} ({d['distinct_6dp']} @6dp){flag}")
    print(f"    ROC-AUC {d['roc']:.4f}   PR-AUC {d['pr']:.4f}")
    return d


# ===================================================================== data
df = pd.read_csv(TEST, sep='\t').reset_index(drop=True)
if not V2:
    # v1 comparison mode only: keep the rows every baseline can encode, so all five
    # still score an identical row set. v2 needs no filtering at all.
    _n0 = len(df)
    df = df[df.cdr3.str.len().le(20) & df['antigen.epitope'].str.len().le(22)
            & df.cdr3.str.fullmatch('[ARNDCQEGHILKMFPSTWYV]+')
            & df['antigen.epitope'].str.fullmatch('[ARNDCQEGHILKMFPSTWYV]+')].reset_index(drop=True)
    print(f"v1 comparison mode: {len(df)}/{_n0} rows retained (CDR3<=20, standard AA)")
TCRS = df['cdr3'].tolist()
PEPS = df['antigen.epitope'].tolist()
Y = df['label'].values.astype(int)
N = len(df)
print("=" * 78)
print(f"R2-3 baseline rerun | {TEST}")
print(f"n = {N}  ({Y.sum()} pos / {(Y == 0).sum()} neg)  "
      f"{df['antigen.epitope'].nunique()} epitopes  seed = {SEED}  device = {DEVICE}")
AAS = set('ARNDCQEGHILKMFPSTWYV')
assert all(set(t) <= AAS for t in TCRS) and all(set(p) <= AAS for p in PEPS), "non-standard AA"
assert max(len(t) for t in TCRS) <= 20 and max(len(p) for p in PEPS) <= 22
print("all rows standard-AA, CDR3 <= 20, epitope <= 22 -> every model scores all "
      f"{N} rows, no filtering, identical row order")
print("=" * 78)

PRED = {}
KEYS = {}
DIST = {}


# ===================================================================== 1. STEP
print("\n[1/6] STEP  (v2-trained 'corrected' checkpoints, 3 seeds)")
step_cols = []
if not V2:
    print("  SKIPPED in v1 comparison mode - the checkpoints_v2 models were trained on v2 "
          "train, so scoring them on the v1 test would not be epitope-disjoint.")
set_seed()
from models.step_model import STEPModel
from models.dataset_physchem_v2 import PhysChemV2Dataset
from torch.utils.data import DataLoader

if V2:
    ds = PhysChemV2Dataset(TEST, verbose=False)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    assert len(ds) == N, "dataset length != tsv length"

for seed in ((42, 43, 44) if V2 else ()):
    ckpt_path = f'models/checkpoints_v2/corrected_seed{seed}/best_model.pt'
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    m = STEPModel(use_structure_priors=True, priors_file='structure_priors_v2.npz')
    print(f"  seed {seed}  (best_val_auc {ck.get('val_auc', ck.get('best_val_auc', '?'))}, "
          f"epoch {ck.get('epoch', '?')})")
    KEYS[f'STEP_seed{seed}'] = strict_load(m, ck['model_state_dict'], f'STEP_seed{seed}')
    m.to(DEVICE).eval()
    out, ylab = [], []
    with torch.no_grad():
        # PhysChemV2Dataset yields (map, global_feats, label, cdr3_len, epi_len);
        # 'corrected' is use_length_mask=False, so the lengths are unused (train_matrix_v2.py:112)
        for ph, gf, lb, _cl, _el in dl:
            out.extend(m(ph.to(DEVICE), gf.to(DEVICE)).cpu().numpy().ravel())
            ylab.extend(lb.numpy().ravel())
    assert np.array_equal(np.asarray(ylab, dtype=int), Y), "STEP dataloader reordered rows"
    p = np.asarray(out, dtype=np.float64)
    DIST[f'STEP_seed{seed}'] = describe(f'STEP_seed{seed}', p, Y)
    PRED[f'STEP_seed{seed}'] = p
    step_cols.append(p)
    del m
if step_cols:
    PRED['STEP_ensemble'] = np.mean(step_cols, axis=0)
    print("  seed-ensemble (mean predicted probability):")
    DIST['STEP_ensemble'] = describe('STEP_ensemble', PRED['STEP_ensemble'], Y)


# ===================================================================== 2. DAISY
print("\n[2/6] DAISY  (unchanged - DAISY_ours.pt, retrained on our data)")
set_seed()
sys.path.insert(0, 'baselines/repos/DAISY')
from DAISY import get_interaction_map, get_global_attributes_diff, resnet_danet

m = resnet_danet()
KEYS['DAISY'] = strict_load(m, torch.load('baselines/repos/DAISY/model/DAISY_ours.pt',
                                          map_location='cpu'), 'DAISY')
m.to(DEVICE).eval()
maps = torch.tensor(get_interaction_map(TCRS, PEPS), dtype=torch.float32)
diffs = torch.tensor(get_global_attributes_diff(TCRS, PEPS), dtype=torch.float32)
out = []
with torch.no_grad():
    for i in range(0, N, 256):
        o = m(maps[i:i + 256].to(DEVICE), diffs[i:i + 256].to(DEVICE))
        out.extend(torch.softmax(o, dim=1)[:, 1].cpu().numpy() if o.dim() > 1
                   else torch.sigmoid(o).cpu().numpy().ravel())
PRED['DAISY'] = np.asarray(out, dtype=np.float64)
assert len(PRED['DAISY']) == N
DIST['DAISY'] = describe('DAISY', PRED['DAISY'], Y)
sys.path.pop(0)
del m


# ===================================================================== 3. ERGO-LSTM
print("\n[3/6] ERGO-LSTM  (unchanged - released lstm_mcpas_memory_protein_model.pt)")
set_seed()
sys.path.insert(0, 'baselines/repos/ERGO')
from ERGO_models import DoubleLSTMClassifier, AutoencoderLSTMClassifier
from evaluation_methods import predict as ergo_predict

m = DoubleLSTMClassifier(10, 30, 0.1, str(DEVICE))
ck = torch.load('baselines/repos/ERGO/models/lstm_mcpas_memory_protein_model.pt',
                map_location='cpu')
KEYS['ERGO-LSTM'] = strict_load(m, ck['model_state_dict'], 'ERGO-LSTM')
m.to(DEVICE).eval()
args_l = argparse.Namespace(model_type='lstm', device=str(DEVICE))
_, _, pr_ = ergo_predict(args_l, m, list(TCRS), list(PEPS))
PRED['ERGO-LSTM'] = np.asarray(pr_, dtype=np.float64).ravel()[:N]
assert len(PRED['ERGO-LSTM']) == N
DIST['ERGO-LSTM'] = describe('ERGO-LSTM', PRED['ERGO-LSTM'], Y)
del m


# ===================================================================== 4. ERGO-AE
print("\n[4/6] ERGO-AE  (FIXED: canonical TCR one-hot index A=0..V=19, X=20)")
print("        submitted script used {aa: i+1}, X=0 -- every column shifted +1")
set_seed()
ck = torch.load('baselines/repos/ERGO/models/ae_mcpas_memory_protein_model.pt',
                map_location='cpu')
ae_only = {'model_state_dict': {k.replace('autoencoder.', ''): v
                                for k, v in ck['model_state_dict'].items()
                                if k.startswith('autoencoder.')}}
tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
torch.save(ae_only, tmp.name)
# batch_size=50 is what evaluation_methods.predict uses for the 'ae' path
m = AutoencoderLSTMClassifier(embedding_dim=10, device=str(DEVICE), max_len=28, input_dim=21,
                              encoding_dim=30, batch_size=50, ae_file=tmp.name, train_ae=False)
KEYS['ERGO-AE'] = strict_load(m, ck['model_state_dict'], 'ERGO-AE')
m.to(DEVICE).eval()
# the authors' own path: evaluation_methods.predict builds tcr_atox canonically
# (evaluation_methods.py:94  {amino: index for index, amino in enumerate(amino_acids + ['X'])})
args_a = argparse.Namespace(model_type='ae', device=str(DEVICE))
_, _, pr_ = ergo_predict(args_a, m, list(TCRS), list(PEPS))
PRED['ERGO-AE'] = np.asarray(pr_, dtype=np.float64).ravel()[:N]
assert len(PRED['ERGO-AE']) == N
DIST['ERGO-AE'] = describe('ERGO-AE', PRED['ERGO-AE'], Y)
os.unlink(tmp.name)
sys.path.pop(0)
del m


# ===================================================================== 5. TEINet
print("\n[5/6] TEINet  (FIXED: real released teinet_data.pth via the authors' own code)")
print("        submitted number came from a randomly initialised STEPModel")
set_seed()
_cwd = os.getcwd()
os.chdir('baselines/repos/TEINet')
sys.path.insert(0, os.getcwd())
from tcrpeg.TCRpeg import TCRpeg
from model import TEINet

en_t = TCRpeg(hidden_size=768, num_layers=3, load_data=False,
              embedding_path='encoders/aa_emb_tcr.txt', device=str(DEVICE))
en_t.create_model()
en_e = TCRpeg(hidden_size=768, num_layers=3, load_data=False,
              embedding_path='encoders/aa_emb_tcr.txt', device=str(DEVICE))
en_e.create_model()
m = TEINet(en_tcr=en_t, en_epi=en_e, cat_size=768 * 2, dropout=0.1, normalize=True,
           weight_decay=0, device=str(DEVICE)).to(DEVICE)
KEYS['TEINet'] = strict_load(m, torch.load('teinet_data.pth', map_location=str(DEVICE)), 'TEINet')
m.eval()
from scipy.special import expit
out = []
with torch.no_grad():
    for i in range(0, N, 128):
        s = m(TCRS[i:i + 128], PEPS[i:i + 128])
        out.extend(expit(s.view(-1).detach().cpu().numpy()))
os.chdir(_cwd)
sys.path.pop(0)
PRED['TEINet'] = np.asarray(out, dtype=np.float64)
assert len(PRED['TEINet']) == N
DIST['TEINet'] = describe('TEINet', PRED['TEINet'], Y)
del m


# ===================================================================== 6. ATM-TCR
print("\n[6/6] ATM-TCR  (FIXED: repo AMINO_MAP, pad index 24, pad_type='mid')")
print("        submitted script end-padded with index 20 = amino acid 'B'")
set_seed()
sys.path.insert(0, 'baselines/repos/ATM-TCR')
from attention import Net

# data_loader.py:11-15
AMINO_MAP = {'<pad>': 24, '*': 23, 'A': 0, 'C': 4, 'B': 20, 'E': 6, 'D': 3, 'G': 7,
             'F': 13, 'I': 9, 'H': 8, 'K': 11, 'M': 12, 'L': 10, 'N': 2, 'Q': 5,
             'P': 14, 'S': 15, 'R': 1, 'T': 16, 'W': 17, 'V': 19, 'Y': 18,
             'X': 22, 'Z': 21}
PAD = AMINO_MAP['<pad>']


def pad_mid(seq, max_len):
    """data_loader.py:400-409 Field_modified.pad with pad_type='mid', no init/eos token."""
    x = list(seq)
    i_gap = int(np.ceil(min(len(x), max_len) / 2))
    i_gap_rev = min(len(x), max_len) - i_gap
    toks = x[:i_gap] + ['<pad>'] * max(0, max_len - len(x)) + (x[-i_gap_rev:] if i_gap_rev > 0 else [])
    return [AMINO_MAP[t] for t in toks]


class _Args:
    pep_length = max_len_pep = 22
    tcr_length = max_len_tcr = 20
    drop_rate = 0.25; lin_size = 1024; heads = 5
    blosum = None; padding = 'mid'; model = 'attention'


# embedding is a plain nn.Embedding when blosum is None (attention.py:10-12); the array
# passed in only fixes its shape and is overwritten by the strict load below.
m = Net(np.zeros((26, 25), dtype=np.float32), _Args()).to(DEVICE)
ck = torch.load('baselines/repos/ATM-TCR/models/original.ckpt', map_location=DEVICE)
KEYS['ATM-TCR'] = strict_load(m, ck, 'ATM-TCR')
assert torch.allclose(m.embedding.weight.detach().cpu(), ck['embedding.weight'].cpu()), \
    "ATM-TCR embedding not taken from checkpoint"
m.eval()
pt = torch.LongTensor([pad_mid(p, 22) for p in PEPS]).to(DEVICE)
tt = torch.LongTensor([pad_mid(t, 20) for t in TCRS]).to(DEVICE)
assert pt.shape == (N, 22) and tt.shape == (N, 20)
out = []
with torch.no_grad():
    for i in range(0, N, 256):
        out.extend(m(pt[i:i + 256], tt[i:i + 256]).cpu().numpy().ravel())
PRED['ATM-TCR'] = np.asarray(out, dtype=np.float64)
assert len(PRED['ATM-TCR']) == N
DIST['ATM-TCR'] = describe('ATM-TCR', PRED['ATM-TCR'], Y)
sys.path.pop(0)
del m


# ===================================================================== save
os.makedirs(OUTDIR, exist_ok=True)
order = [k for k in ['STEP_seed42', 'STEP_seed43', 'STEP_seed44', 'STEP_ensemble',
                     'DAISY', 'ERGO-LSTM', 'ERGO-AE', 'TEINet', 'ATM-TCR'] if k in PRED]
for k in order:
    assert len(PRED[k]) == N, k

np.savez_compressed(f'{OUTDIR}/test_predictions_{"v2" if V2 else "v1check"}.npz',
                    y=Y, cdr3=np.array(TCRS), epitope=np.array(PEPS),
                    **{k: PRED[k] for k in order})

out_df = pd.DataFrame({'cdr3': TCRS, 'antigen.epitope': PEPS, 'label': Y})
for k in order:
    out_df[k] = PRED[k]
out_df.to_csv(f'{OUTDIR}/per_sample_predictions_{"v2" if V2 else "v1check"}.csv', index=False)

summary = []
for k in order:
    d = DIST[k]
    summary.append(dict(model=k, n=d['n'], roc_auc=d['roc'], pr_auc=d['pr'],
                        pred_min=d['min'], pred_max=d['max'], pred_mean=d['mean'],
                        pred_sd=d['sd'], distinct=d['distinct'], distinct_6dp=d['distinct_6dp'],
                        keys_expected=KEYS.get(k, {}).get('expects'),
                        keys_loaded=KEYS.get(k, {}).get('loaded')))
pd.DataFrame(summary).to_csv(f'{OUTDIR}/summary_{"v2" if V2 else "v1check"}.csv', index=False)
json.dump({'test_file': TEST, 'n': N, 'seed': SEED, 'keys': KEYS, 'dist': DIST},
          open(f'{OUTDIR}/rerun_report_{"v2" if V2 else "v1check"}.json', 'w'), indent=2, default=str)

print("\n" + "=" * 78)
print(pd.DataFrame(summary)[['model', 'roc_auc', 'pr_auc', 'pred_sd',
                             'distinct_6dp', 'keys_expected', 'keys_loaded']].to_string(index=False))
print("=" * 78)
_tag = 'v2' if V2 else 'v1check'
print(f"saved: {OUTDIR}/per_sample_predictions_{_tag}.csv  (row order = {TEST})")
print(f"       {OUTDIR}/test_predictions_{_tag}.npz")
print(f"       {OUTDIR}/summary_{_tag}.csv")
