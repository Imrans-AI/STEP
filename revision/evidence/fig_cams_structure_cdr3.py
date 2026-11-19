"""
Recompute the attention maps using each STRUCTURE's own CDR3b, not the CDR3b the
submitted figures used.

results/fig6_scorecam/metadata.csv pairs each epitope with a CDR3b that is NOT
the one in the crystal structure the panel projects onto:

   3GSN  structure ASSPVTGGIYGYT   figure used CASSKTQLWETQYF
   6RSY  structure ASSLGF-GRDVMR   figure used ASSLGGNGYT
   1OGA  structure ASSSRS--SYEQY   figure used CASSIISQNEQFF
   4QOK  structure AWSETGLGTGELF   figure used CAWSETGLGTGELFF  (same loop,
                                   ANARCI trims the flanking C and F)

For the first three the panels therefore show attention for one receptor drawn
on another receptor's structure - the same class of defect as the 5HHO/4QOK
mix-up already found in Fig 5c. Contact support can only be read off a map whose
CDR3b is the structure's own, so the maps are recomputed on that basis here.

PDB residue numbering is read from the structure, not assumed.
"""
import ast
import json
import pickle
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
SEEDS = [42, 43, 44]
AA3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
       'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
       'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
       'Y': 'TYR', 'V': 'VAL'}
THREE2ONE = {v: k for k, v in AA3.items()}
NAMES = ['HYDROPHOBICITY', 'CHARGE', 'FLEXIBILITY', 'REFRACTIVITY', 'AREA_LOSS']
_t = ast.parse(open(BASE / 'data/compute_physchem.py').read())
TAB = {n.targets[0].id: ast.literal_eval(n.value) for n in _t.body
       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
       and n.targets[0].id in NAMES}
PROPS = [TAB[n] for n in NAMES]


def chains(pdb_path):
    out = {}
    for line in open(pdb_path):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            c, r, n = line[21], int(line[22:26]), line[17:20].strip()
            out.setdefault(c, []).append((r, THREE2ONE.get(n, 'X')))
    return {c: v for c, v in out.items()}


def find_numbering(pdb_path, seq):
    """Residue numbers of `seq` wherever it occurs in any chain."""
    for c, res in chains(pdb_path).items():
        s = ''.join(x[1] for x in res)
        i = s.find(seq)
        if i != -1:
            return c, [res[k][0] for k in range(i, i + len(seq))]
    return None, None


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


contacts = pickle.load(open(BASE / 'data/raw/stcrdab_cdr3_contacts.pkl', 'rb'))
by_pdb = {}
for e in contacts:
    by_pdb.setdefault(str(e['pdb']).lower(), []).append(e)

JOBS = [('3GSN', 'NLVPMVATV', 'Fig 5a,b'), ('6RSY', 'RMFPNAPYL', 'Fig S2a'),
        ('1OGA', 'GILGFVFTL', 'Fig S2b'), ('4QOK', 'EAAGIGILTV', 'Fig 5c,d / Fig 6')]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rep = {}
for pdb, epi, where in JOBS:
    ent = by_pdb[pdb.lower()][0]
    cdr3 = ent['cdr3b_seq'].replace('-', '')
    assert ent['epitope_seq'] == epi, (pdb, ent['epitope_seq'], epi)
    dm = np.array(ent['cdr3_epitope_distance'], dtype=float)
    assert dm.shape == (len(cdr3), len(epi)), (pdb, dm.shape, len(cdr3), len(epi))
    ch, nums = find_numbering(BASE / f'data/pdb_fig6/{pdb}.pdb', cdr3)
    echain, enums = find_numbering(BASE / f'data/pdb_fig6/{pdb}.pdb', epi)

    probs, cams = [], []
    for s in SEEDS:
        p, c = cam_for(cdr3, epi, s, device)
        probs.append(p); cams.append(c)
    mean = np.stack(cams).mean(0)
    mean = mean / (mean.max() + 1e-9)
    np.save(OUT / f'{pdb}_structure_cam_v2.npy', mean)
    t = mean[:len(cdr3), :len(epi)]
    cc = dm < 8.0
    eo, ro = np.argsort(-t.sum(0)), np.argsort(-t.sum(1))
    lab_e = (lambda j: f'{AA3[epi[j]]}-{enums[j] if enums else j + 1}')
    lab_r = (lambda i: f'{AA3[cdr3[i]]}-{nums[i] if nums else i + 1}')
    etop, rtop = [lab_e(j) for j in eo[:3]], [lab_r(i) for i in ro[:3]]
    esup = [lab_e(j) for j in eo[:3] if cc[:, j].any()]
    rsup = [lab_r(i) for i in ro[:3] if cc[i].any()]
    print(f'\n{pdb} / {epi}   ({where})')
    print(f'  structure CDR3b {cdr3} on chain {ch}, residues '
          f'{nums[0] if nums else "?"}-{nums[-1] if nums else "?"}; '
          f'epitope chain {echain}')
    print(f'  P(bind) {np.mean(probs):.4f} +/- {np.std(probs, ddof=1):.4f}')
    print(f'  epitope top 3: {", ".join(etop)}')
    print(f'  CDR3b   top 3: {", ".join(rtop)}')
    print(f'  with a true Ca-Ca contact < 8 A: epitope {len(esup)}/3 {esup}; '
          f'CDR3b {len(rsup)}/3 {rsup}')
    print(f'  contacts in the matrix: {int(cc.sum())} of {cc.size}')
    rep[pdb] = dict(epitope=epi, where=where, cdr3=cdr3, chain=ch,
                    resnums=[int(x) for x in nums] if nums else None,
                    p_mean=float(np.mean(probs)), p_sd=float(np.std(probs, ddof=1)),
                    p_seeds=probs, epitope_top3=etop, cdr3_top3=rtop,
                    epitope_with_contact=esup, cdr3_with_contact=rsup,
                    n_contacts=int(cc.sum()))

json.dump(rep, open(OUT / 'cams_structure_cdr3.json', 'w'), indent=2, default=str)
print('\nsaved', OUT / 'cams_structure_cdr3.json')
