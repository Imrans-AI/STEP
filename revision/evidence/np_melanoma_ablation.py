"""
Numbers pass, part D: the v2 melanoma (and lung) ablation — inference only.

Closes [[PENDING: R2-7]]. The fifteen checkpoints in models/checkpoints_v2/ are
scored on the cancer sets; nothing is trained.

Why a local dataset rather than PhysChemV2Dataset: physchem_index_v2.json was
built from the v2 splits pool only, so 826 of 1,582 melanoma pairs and 113 of
170 lung pairs are absent from it and PhysChemV2Dataset raises KeyError. The
maps are therefore computed here, from the same five tables and the same
abs(a-b) formula. The tables are PARSED out of data/compute_physchem.py with
ast, never imported: importing that file runs its main body, which writes
embeddings/ relative to the working directory (see R2-5_cancer_memorisation.md).

Correctness is not assumed. For every cancer pair that IS present in
physchem_index_v2.json the locally computed map is asserted bit-equal to the
stored one before any model runs.

Nothing under data/, models/ or results/ is written.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
sys.path.insert(0, str(BASE / 'models'))
from step_model import STEPModel                          # noqa: E402
from dataset_physchem_only import compute_global_features  # noqa: E402
from prior_variants import get_prior, VARIANTS            # noqa: E402

EMB = BASE / 'data' / 'embeddings'
CKPT = BASE / 'models' / 'checkpoints_v2'
OUT = BASE / 'revision' / 'evidence' / 'numbers_pass'
SEEDS = [42, 43, 44]
MAX_CDR3, MAX_EPI = 20, 15

NAMES = ['HYDROPHOBICITY', 'CHARGE', 'FLEXIBILITY', 'REFRACTIVITY', 'AREA_LOSS']
_tree = ast.parse(open(BASE / 'data/compute_physchem.py').read())
TABLES = {n.targets[0].id: ast.literal_eval(n.value) for n in _tree.body
          if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
          and n.targets[0].id in NAMES}
assert len(TABLES) == 5 and all(len(v) == 20 for v in TABLES.values())
PROPS = [TABLES[n] for n in NAMES]          # model channel order


def physchem_map(cdr3, epi):
    m = np.zeros((5, MAX_CDR3, MAX_EPI), dtype=np.float32)
    for k, tbl in enumerate(PROPS):
        for i, a in enumerate(cdr3[:MAX_CDR3]):
            for j, b in enumerate(epi[:MAX_EPI]):
                if a in tbl and b in tbl:
                    m[k, i, j] = abs(tbl[a] - tbl[b])
    return m


class CancerDataset(Dataset):
    def __init__(self, path, verify=True):
        self.data = pd.read_csv(path, sep='\t')
        c = self.data.cdr3.astype(str).values
        e = self.data['antigen.epitope'].astype(str).values
        self.maps = np.stack([physchem_map(x, y) for x, y in zip(c, e)])
        self.gf = np.array([compute_global_features(x, y) for x, y in zip(c, e)],
                           dtype=np.float32)
        self.labels = self.data['label'].to_numpy()
        self.cdr3_len = np.array([min(len(x), MAX_CDR3) for x in c], dtype=np.int64)
        self.epi_len = np.array([min(len(y), MAX_EPI) for y in e], dtype=np.int64)
        if verify:
            self._verify(c, e)

    def _verify(self, c, e):
        idx = json.load(open(EMB / 'physchem_index_v2.json'))
        main = np.load(EMB / 'physchem_maps.npy', mmap_mode='r')
        n_orig = main.shape[0]
        extra_p = EMB / 'physchem_maps_v2_extra.npy'
        extra = np.load(extra_p, mmap_mode='r') if extra_p.exists() else None
        checked = worst = 0
        for i, (x, y) in enumerate(zip(c, e)):
            k = f'{x}|{y}'
            if k not in idx:
                continue
            j = idx[k]
            stored = np.asarray(main[j] if j < n_orig else extra[j - n_orig])
            worst = max(worst, float(np.abs(stored - self.maps[i]).max()))
            checked += 1
        print(f'    verified {checked} pairs against the stored maps; '
              f'max abs difference {worst:.3e}', flush=True)
        assert worst == 0.0, 'locally computed maps differ from the stored ones'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return (torch.from_numpy(self.maps[i]).float(),
                torch.from_numpy(self.gf[i]).float(),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.cdr3_len[i], dtype=torch.long),
                torch.tensor(self.epi_len[i], dtype=torch.long))


def predict(ckpt_dir, prior, seed, loader, device):
    _, use_mask = get_prior(prior, seed)
    model = STEPModel(use_structure_priors=True, use_length_mask=use_mask,
                      priors_file='structure_priors_v2.npz').to(device)
    sd = torch.load(ckpt_dir / 'best_model.pt', map_location=device,
                    weights_only=False)['model_state_dict']
    model.load_state_dict(sd, strict=True)
    model.eval()
    out = []
    with torch.no_grad():
        for phys, gf, y, cl, el in loader:
            phys, gf, cl, el = phys.to(device), gf.to(device), cl.to(device), el.to(device)
            o = model(phys, gf, cdr3_len=cl, epi_len=el) if use_mask else model(phys, gf)
            out.append(o.cpu().numpy())
    return np.concatenate(out)


def balanced_subset(mask, y, rng):
    """down-sample the majority class inside `mask`, as the submitted analysis did"""
    pos = np.where(mask & (y == 1))[0]
    neg = np.where(mask & (y == 0))[0]
    n = min(len(pos), len(neg))
    if n == 0:
        return None
    return np.concatenate([rng.choice(pos, n, replace=False),
                           rng.choice(neg, n, replace=False)])


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device, flush=True)

    tr = pd.read_csv(BASE / 'data/splits_600_epitope_v2/train.tsv', sep='\t')
    train_pairs = set(zip(tr.cdr3.astype(str), tr['antigen.epitope'].astype(str)))
    print(f'v2 train pairs: {len(train_pairs):,}', flush=True)

    rows, store = [], {}
    for tag, path in [('melanoma', 'data/cancer_specific/melanoma_test.tsv'),
                      ('lung', 'data/cancer_specific/lung_test.tsv')]:
        print(f'\n=== {tag} ===', flush=True)
        ds = CancerDataset(BASE / path)
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
        y = ds.labels.astype(int)
        pairs = list(zip(ds.data.cdr3.astype(str), ds.data['antigen.epitope'].astype(str)))
        free = np.array([p not in train_pairs for p in pairs])
        rng = np.random.default_rng(42)
        bal = balanced_subset(free, y, rng)
        print(f'  n={len(y)} pos={y.sum()} | overlap-free n={free.sum()} '
              f'(pos {int(y[free].sum())}) | balanced overlap-free n='
              f'{0 if bal is None else len(bal)}', flush=True)

        for v in VARIANTS:
            for s in SEEDS:
                d = CKPT / f'{v}_seed{s}'
                if not (d / 'results.json').exists():
                    continue
                p = predict(d, v, s, loader, device)
                store[(tag, v, s)] = p
                r = dict(set=tag, prior=v, seed=s,
                         n=len(y), roc=roc_auc_score(y, p),
                         pr=average_precision_score(y, p))
                if bal is not None and len(np.unique(y[bal])) == 2:
                    r['roc_free_bal'] = roc_auc_score(y[bal], p[bal])
                    r['n_free_bal'] = len(bal)
                else:
                    r['roc_free_bal'] = np.nan
                    r['n_free_bal'] = 0 if bal is None else len(bal)
                rows.append(r)
                print(f"  {v:12s} s{s}: ROC {r['roc']:.4f}  PR {r['pr']:.4f}  "
                      f"overlap-free balanced ROC {r['roc_free_bal']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'cancer_ablation_v2.csv', index=False)
    np.savez_compressed(OUT / 'cancer_ablation_preds.npz',
                        **{f'{t}_{v}_s{s}': p for (t, v, s), p in store.items()})

    print('\n=== mean +/- SD over three seeds ===', flush=True)
    for tag in ('melanoma', 'lung'):
        print(f'\n  {tag}')
        for v in VARIANTS:
            g = df[(df.set == tag) & (df.prior == v)]
            if not len(g):
                continue
            print(f"    {v:12s} ROC {g.roc.mean():.4f}+/-{g.roc.std(ddof=1):.4f}  "
                  f"PR {g.pr.mean():.4f}+/-{g.pr.std(ddof=1):.4f}  "
                  f"overlap-free bal ROC {g.roc_free_bal.mean():.4f}"
                  f"+/-{g.roc_free_bal.std(ddof=1):.4f} (n={int(g.n_free_bal.iloc[0])})",
                  flush=True)
    print('\nsaved', OUT / 'cancer_ablation_v2.csv', flush=True)


if __name__ == '__main__':
    main()
