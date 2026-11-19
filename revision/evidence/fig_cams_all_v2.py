"""
Recompute the remaining attention maps on the v2 checkpoints (Fig 5a,b; Fig 6;
Fig S2 a,b), and answer R1-4: are the visualisation structures among the 198
used to build the prior?

Same method as the Fig 5 rebuild: gradient x input on the physicochemical map,
summed over channels, ReLU'd, max-normalised, averaged over three seeds. Tables
parsed with ast, never imported.

True contacts come from data/raw/stcrdab_cdr3_contacts.pkl, which already holds
a cdr3_epitope_distance matrix per structure - no PDB parsing needed, and it is
the same source the prior was built from.
"""
import ast
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
sys.path.insert(0, str(BASE / 'models'))
from step_model import STEPModel                           # noqa: E402
from dataset_physchem_only import compute_global_features  # noqa: E402
from prior_variants import get_prior                       # noqa: E402

OUT = BASE / 'revision/evidence/figures'
SEEDS = [42, 43, 44]
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

contacts = pickle.load(open(BASE / 'data/raw/stcrdab_cdr3_contacts.pkl', 'rb'))
by_pdb = {}
for e in contacts:
    by_pdb.setdefault(str(e['pdb']).lower(), []).append(e)

# ---------------------------------------------------------------- R1-4
print('=' * 76); print('  R1-4  are the visualisation structures in the 198?'); print('=' * 76)
tsv = pd.read_csv(BASE / 'data/raw/stcrdab_human_mhci_filtered.tsv', sep='\t')
r14 = {}
for p in ('3gsn', '4qok', '6rsy', '1oga', '3o4l', '5hho'):
    n_tsv = int((tsv.pdb.str.lower() == p).sum())
    n_pkl = len(by_pdb.get(p, []))
    r14[p] = dict(in_filtered_tsv=n_tsv, in_contacts_pkl=n_pkl)
    print(f'  {p.upper()}: {n_tsv} row(s) in the filtered table, '
          f'{n_pkl} entr(y/ies) in the contact pickle  -> '
          f'{"INCLUDED in the prior" if n_pkl else "not included"}')
n_used = sum(len(v) for v in by_pdb.values())
print(f'\n  198 structures total; the four figure structures contribute '
      f'{sum(len(by_pdb.get(p, [])) for p in ("3gsn", "4qok", "6rsy", "1oga"))} '
      f'of them.')

# cost of a leave-those-out prior: rebuild it and measure the change
OFFENDERS = {'3gsn', '4qok', '6rsy', '1oga'}


def build_prior(entries):
    acc, n = np.zeros((20, 15)), 0
    for d in entries:
        m = d.get('cdr3_epitope_distance')
        if not isinstance(m, np.ndarray) or m.ndim != 2:
            continue
        padded = np.zeros((20, 15))
        h, w = m.shape
        padded[:min(h, 20), :min(w, 15)] = m[:min(h, 20), :min(w, 15)]
        acc += (padded < 6.0).astype(float)
        n += 1
    avg = acc / n
    wgt = avg + 0.1
    return wgt / wgt.max(), n


full_w, n_full = build_prior(contacts)
loo_w, n_loo = build_prior([e for e in contacts
                            if str(e['pdb']).lower() not in OFFENDERS])
delta = np.abs(full_w - loo_w)
print(f'\n  leave-those-out prior: {n_loo} structures instead of {n_full}')
print(f'  max |change| in a cell {delta.max():.4f}, mean {delta.mean():.4f}, '
      f'correlation {np.corrcoef(full_w.ravel(), loo_w.ravel())[0, 1]:.6f}')
print('  cost: rebuilding the prior is seconds of numpy. Using it means '
      'retraining,\n  which is 3 seeds x the usual run - the same unit of work '
      'as one row of\n  the prior-control matrix.')
r14['leave_out'] = dict(n_full=n_full, n_loo=n_loo, max_delta=float(delta.max()),
                        mean_delta=float(delta.mean()),
                        corr=float(np.corrcoef(full_w.ravel(), loo_w.ravel())[0, 1]))

# ---------------------------------------------------------------- CAMs
print('\n' + '=' * 76); print('  ATTENTION MAPS RECOMPUTED ON v2'); print('=' * 76)


def pmap(c, e):
    m = np.zeros((5, 20, 15), dtype=np.float32)
    for k, tbl in enumerate(PROPS):
        for i, a in enumerate(c[:20]):
            for j, b in enumerate(e[:15]):
                if a in tbl and b in tbl:
                    m[k, i, j] = abs(tbl[a] - tbl[b])
    return m


def cam_for(cdr3, epi, seed, device):
    _, use_mask = get_prior('corrected', seed)
    m = STEPModel(use_structure_priors=True, use_length_mask=use_mask,
                  priors_file='structure_priors_v2.npz').to(device)
    sd = torch.load(BASE / f'models/checkpoints_v2/corrected_seed{seed}/best_model.pt',
                    map_location=device, weights_only=False)['model_state_dict']
    m.load_state_dict(sd, strict=True)
    m.eval()
    pm = torch.from_numpy(pmap(cdr3, epi)[None]).float().to(device).requires_grad_(True)
    gf = torch.from_numpy(compute_global_features(cdr3, epi)[None]).float().to(device)
    cl = torch.tensor([min(len(cdr3), 20)]).to(device)
    el = torch.tensor([min(len(epi), 15)]).to(device)
    out = m(pm, gf, cdr3_len=cl, epi_len=el) if use_mask else m(pm, gf)
    p = float(out.item())
    m.zero_grad(); out.backward()
    c = (pm.grad * pm).sum(1).squeeze().detach().cpu().numpy()
    c = np.maximum(c, 0)
    return p, c / (c.max() + 1e-9)


PAIRS = [
    ('NLVPMVATV', 'CASSKTQLWETQYF', '3gsn', 95, 'NLVPMVATV_cam.npy', 'Fig 5a,b'),
    ('RMFPNAPYL', 'ASSLGGNGYT', '6rsy', 95, 'RMFPNAPYL_cam.npy', 'Fig S2a'),
    ('GILGFVFTL', 'CASSIISQNEQFF', '3o4l', 95, 'GILGFVFTL_cam.npy', 'Fig S2b'),
]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rep = {'r14': r14, 'cams': {}}
for epi, cdr3, pdb, off, oldf, where in PAIRS:
    probs, cams = [], []
    for s in SEEDS:
        p, c = cam_for(cdr3, epi, s, device)
        probs.append(p); cams.append(c)
    mean = np.stack(cams).mean(0)
    mean = mean / (mean.max() + 1e-9)
    np.save(OUT / f'{epi}_{pdb.upper()}_cam_v2.npy', mean)
    nc, ne = len(cdr3), len(epi)
    t = mean[:nc, :ne]
    eo, ro = np.argsort(-t.sum(0)), np.argsort(-t.sum(1))
    etop = [f'{AA3[epi[j]]}-{j + 1}' for j in eo[:3]]
    rtop = [f'{AA3[cdr3[i]]}-{i + off - 1 + 1}' for i in ro[:3]]
    old = np.load(BASE / f'results/fig6_scorecam/{oldf}')[:nc, :ne]
    oe = [f'{AA3[epi[j]]}-{j + 1}' for j in np.argsort(-old.sum(0))[:3]]
    orr = [f'{AA3[cdr3[i]]}-{i + off}' for i in np.argsort(-old.sum(1))[:3]]

    ent = by_pdb.get(pdb, [])
    cc = None
    if ent:
        dm = ent[0]['cdr3_epitope_distance']
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            cc = dm[:nc, :ne] < 8.0
    sup = [e for k, e in zip(eo[:3], etop) if cc is not None and k < cc.shape[1]
           and cc[:, k].any()]
    rsup = [r for k, r in zip(ro[:3], rtop) if cc is not None and k < cc.shape[0]
            and cc[k].any()]
    corr = float(np.corrcoef(t.ravel(), old.ravel())[0, 1])
    print(f'\n  {epi} / {pdb.upper()}  ({where})  CDR3b {cdr3}')
    print(f'    P(bind) {np.mean(probs):.4f} +/- {np.std(probs, ddof=1):.4f}  '
          f'seeds {", ".join(f"{p:.3f}" for p in probs)}')
    print(f'    epitope  old {oe}  ->  new {etop}')
    print(f'    CDR3b    old {orr}  ->  new {rtop}')
    print(f'    r(new, old) = {corr:+.3f}')
    print(f'    contact support: epitope {sup or "none"} ; CDR3b {rsup or "none"}'
          + ('' if cc is not None else '  (no contact matrix)'))
    rep['cams'][epi] = dict(pdb=pdb.upper(), cdr3=cdr3, where=where,
                            p_seeds=probs, p_mean=float(np.mean(probs)),
                            p_sd=float(np.std(probs, ddof=1)),
                            epitope_old=oe, epitope_new=etop,
                            cdr3_old=orr, cdr3_new=rtop,
                            corr_with_old=corr,
                            epitope_with_contact=sup, cdr3_with_contact=rsup)

json.dump(rep, open(OUT / 'cams_all_v2.json', 'w'), indent=2, default=str)
print('\nsaved', OUT / 'cams_all_v2.json')
