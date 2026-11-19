"""
Recompute the 4QOK CAM on the v2 checkpoints, and report the residue triplets.

SCORECAM_4QOK.py produced the stored EAAGIGILTV_4QOK_cam.npy from the SUBMITTED
checkpoint. Two things about it are carried over deliberately and two are not:

  * carried over: the attribution method, which despite the file name is
    gradient x input on the physicochemical map, summed over channels, ReLU'd
    and max-normalised - not Score-CAM. The manuscript calls it Score-CAM; that
    discrepancy is recorded in FIGURES_REGENERATED.md, not silently fixed.
  * NOT carried over: `sys.path.insert(0,'data'); from compute_physchem import
    compute_physchem_map`, which executes that file's main body and writes
    embeddings/ relative to the working directory. The tables are parsed with
    ast here instead.

Read-only on data/, models/ and results/. Writes under revision/evidence/.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
sys.path.insert(0, str(BASE / 'models'))
from step_model import STEPModel                           # noqa: E402
from dataset_physchem_only import compute_global_features  # noqa: E402
from prior_variants import get_prior                       # noqa: E402

OUT = BASE / 'revision/evidence/figures'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44]
CDR3, EPI, PDB = 'CAWSETGLGTGELFF', 'EAAGIGILTV', '4QOK'

AA3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
       'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
       'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
       'Y': 'TYR', 'V': 'VAL'}
NAMES = ['HYDROPHOBICITY', 'CHARGE', 'FLEXIBILITY', 'REFRACTIVITY', 'AREA_LOSS']
_t = ast.parse(open(BASE / 'data/compute_physchem.py').read())
TAB = {n.targets[0].id: ast.literal_eval(n.value) for n in _t.body
       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
       and n.targets[0].id in NAMES}
PROPS = [TAB[n] for n in NAMES]


def pmap(c, e):
    m = np.zeros((5, 20, 15), dtype=np.float32)
    for k, tbl in enumerate(PROPS):
        for i, a in enumerate(c[:20]):
            for j, b in enumerate(e[:15]):
                if a in tbl and b in tbl:
                    m[k, i, j] = abs(tbl[a] - tbl[b])
    return m


def cam_for(seed, device):
    _, use_mask = get_prior('corrected', seed)
    m = STEPModel(use_structure_priors=True, use_length_mask=use_mask,
                  priors_file='structure_priors_v2.npz').to(device)
    sd = torch.load(BASE / f'models/checkpoints_v2/corrected_seed{seed}/best_model.pt',
                    map_location=device, weights_only=False)['model_state_dict']
    m.load_state_dict(sd, strict=True)
    m.eval()
    pm = torch.from_numpy(pmap(CDR3, EPI)[None]).float().to(device).requires_grad_(True)
    gf = torch.from_numpy(compute_global_features(CDR3, EPI)[None]).float().to(device)
    cl = torch.tensor([min(len(CDR3), 20)]).to(device)
    el = torch.tensor([min(len(EPI), 15)]).to(device)
    out = m(pm, gf, cdr3_len=cl, epi_len=el) if use_mask else m(pm, gf)
    p = float(out.item())
    m.zero_grad()
    out.backward()
    cam = (pm.grad * pm).sum(1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-9)
    return p, cam


device = 'cuda' if torch.cuda.is_available() else 'cpu'
probs, cams = [], []
for s in SEEDS:
    p, c = cam_for(s, device)
    probs.append(p)
    cams.append(c)
    print(f'  seed {s}: P(bind) {p:.4f}', flush=True)
cams = np.stack(cams)
mean_cam = cams.mean(0)
mean_cam = mean_cam / (mean_cam.max() + 1e-9)
np.save(OUT / 'EAAGIGILTV_4QOK_cam_v2.npy', mean_cam)
np.save(OUT / 'EAAGIGILTV_4QOK_cam_v2_per_seed.npy', cams)

nc, ne = len(CDR3), len(EPI)
t = mean_cam[:nc, :ne]
ecol, rrow = t.sum(0), t.sum(1)
eo, ro = np.argsort(-ecol), np.argsort(-rrow)
etop = [f'{AA3[EPI[j]]}-{j + 1}' for j in eo[:3]]
rtop = [f'{AA3[CDR3[i]]}-{i + 1}' for i in ro[:3]]

print(f'\n  trimmed CAM {t.shape} (CDR3b {nc} x epitope {ne})')
print('\n  EPITOPE positions, share of column attention:')
for j in np.argsort(-ecol):
    print(f'    {AA3[EPI[j]]}-{j + 1:<3} {100 * ecol[j] / ecol.sum():>6.1f}%')
print('\n  CDR3b positions, share of row attention:')
for i in np.argsort(-rrow):
    print(f'    {AA3[CDR3[i]]}-{i + 1:<3} {100 * rrow[i] / rrow.sum():>6.1f}%')
print(f'\n  TOP 3 epitope: {", ".join(etop)}')
print(f'  TOP 3 CDR3b  : {", ".join(rtop)}')

# per-seed stability of the triplets
per = []
for k, s in enumerate(SEEDS):
    tt = cams[k][:nc, :ne]
    e3 = [f'{AA3[EPI[j]]}-{j + 1}' for j in np.argsort(-tt.sum(0))[:3]]
    r3 = [f'{AA3[CDR3[i]]}-{i + 1}' for i in np.argsort(-tt.sum(1))[:3]]
    per.append(dict(seed=s, epitope_top3=e3, cdr3_top3=r3))
    print(f'  seed {s}: epitope {e3}  CDR3b {r3}')

# how the rebuilt CAM compares with the stored one
old = np.load(BASE / 'results/fig6_scorecam/EAAGIGILTV_4QOK_cam.npy')[:nc, :ne]
oe = [f'{AA3[EPI[j]]}-{j + 1}' for j in np.argsort(-old.sum(0))[:3]]
orr = [f'{AA3[CDR3[i]]}-{i + 1}' for i in np.argsort(-old.sum(1))[:3]]
corr = float(np.corrcoef(t.ravel(), old.ravel())[0, 1])
print(f'\n  stored (submitted-checkpoint) CAM: epitope {oe}  CDR3b {orr}')
print(f'  Pearson r between rebuilt and stored CAM: {corr:+.4f}')

# true contacts, for the caption
dist = np.load(BASE / 'results/fig6_scorecam/EAAGIGILTV_4QOK_actual_dist.npy')[:nc, :ne]
cc = dist < 8.0
hit = [AA3[EPI[j]] + f'-{j + 1}' for j in eo[:3] if cc[:, j].any()]
print(f'  of the top-3 epitope positions, those with a true <8 A contact: '
      f'{hit or "none"}')

json.dump(dict(pairs=dict(cdr3=CDR3, epitope=EPI, pdb=PDB),
               p_bind=dict(seeds=probs, mean=float(np.mean(probs)),
                           sd=float(np.std(probs, ddof=1))),
               epitope_top3=etop, cdr3_top3=rtop, per_seed=per,
               stored_cam_top3=dict(epitope=oe, cdr3=orr),
               pearson_vs_stored=corr,
               epitope_share={f'{AA3[EPI[j]]}-{j+1}': float(ecol[j] / ecol.sum())
                              for j in range(ne)},
               cdr3_share={f'{AA3[CDR3[i]]}-{i+1}': float(rrow[i] / rrow.sum())
                           for i in range(nc)}),
          open(OUT / 'cam_4qok_v2.json', 'w'), indent=2)
print('\nsaved', OUT / 'cam_4qok_v2.json')
