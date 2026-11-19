"""
R1-1a / R1-3: where STEP's physicochemical scales come from.

Read-only. Nothing outside revision/evidence/ is written, and no table in the
repository is modified. The five STEP tables are re-read from
data/compute_physchem.py by parsing the source, NOT by importing it: importing
that file runs its main body, which writes embeddings/ relative to the current
directory.
"""
import ast
import json
import re
import subprocess

import numpy as np

ROOT = '/home/muhammad-imran/PhD_data/TCR_Module_1'
AAS = 'ACDEFGHIKLMNPQRSTVWY'
rep = {}
sep = '=' * 78


def parse_dicts(path, names):
    """Pull literal dict assignments out of a source file without executing it."""
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if n in names:
                out[n] = ast.literal_eval(node.value)
    return out


STEP = parse_dicts(f'{ROOT}/data/compute_physchem.py',
                   {'HYDROPHOBICITY', 'CHARGE', 'FLEXIBILITY',
                    'REFRACTIVITY', 'AREA_LOSS'})
assert len(STEP) == 5 and all(len(v) == 20 for v in STEP.values())

# DAISY's own table, read the same way
d_tree = ast.parse(open(f'{ROOT}/baselines/repos/DAISY/DAISY.py').read())
DAISY_RAW = None
for node in d_tree.body:
    if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'amino_acid_properties'):
        def _num(a):
            # '+1.0' parses as UnaryOp(UAdd) and '-1.0' as UnaryOp(USub);
            # treating every UnaryOp as a negation silently flips the sign of
            # every explicitly-plus value, which is most of the charge column.
            if isinstance(a, ast.UnaryOp):
                v = a.operand.value
                return -v if isinstance(a.op, ast.USub) else v
            return a.value
        DAISY_RAW = {k.value: [_num(a) for a in v.args[0].elts]
                     for k, v in zip(node.value.keys, node.value.values)}

assert DAISY_RAW is not None and len(DAISY_RAW) == 20
assert [DAISY_RAW[a][2] for a in 'DEKRH'] == [-1.0, -1.0, 1.0, 1.0, 0.1], \
    f"charge column misparsed: {[DAISY_RAW[a][2] for a in 'DEKRH']}"
DAISY_COLS = ['hydrophobicity', 'refractivity', 'charge', 'flexibility',
              'area_loss']


def vec(d):
    return np.array([d[a] for a in AAS], float)


def minmax(v):
    return (v - v.min()) / (v.max() - v.min())


print(sep); print('  1. THE FIVE STEP TABLES (data/compute_physchem.py)'); print(sep)
for name in ('HYDROPHOBICITY', 'CHARGE', 'FLEXIBILITY', 'REFRACTIVITY', 'AREA_LOSS'):
    v = vec(STEP[name])
    print(f'\n  {name}  min {v.min():.2f} max {v.max():.2f} '
          f'argmin {AAS[int(v.argmin())]} argmax {AAS[int(v.argmax())]} '
          f'distinct {len(set(v))}')
    print('    ' + '  '.join(f'{a}:{STEP[name][a]:.2f}' for a in AAS[:10]))
    print('    ' + '  '.join(f'{a}:{STEP[name][a]:.2f}' for a in AAS[10:]))
rep['step_tables'] = {k: {a: float(v[a]) for a in AAS} for k, v in STEP.items()}

print('\n' + sep)
print('  2. DAISY, RAW AND MIN-MAX NORMALISED (baselines/repos/DAISY/DAISY.py)')
print(sep)
daisy = {}
for i, col in enumerate(DAISY_COLS):
    raw = np.array([DAISY_RAW[a][i] for a in AAS], float)
    daisy[col] = dict(raw=raw, norm=minmax(raw))
    print(f'\n  {col} raw:  ' + '  '.join(f'{a}:{r:g}' for a, r in
                                          zip(AAS[:10], raw[:10])) + ' ...')
    print(f'  {col} norm: ' + '  '.join(f'{a}:{r:.2f}' for a, r in
                                        zip(AAS, daisy[col]["norm"]))[:160])
rep['daisy_raw'] = {a: list(map(float, DAISY_RAW[a])) for a in AAS}

print('\n' + sep)
print('  3. STEP vs DAISY, channel by channel')
print(sep)
pairs = [('HYDROPHOBICITY', 'hydrophobicity'), ('CHARGE', 'charge'),
         ('FLEXIBILITY', 'flexibility'), ('REFRACTIVITY', 'refractivity'),
         ('AREA_LOSS', 'area_loss')]
rep['step_vs_daisy'] = {}
for s, dcol in pairs:
    sv, dv, dr = vec(STEP[s]), daisy[dcol]['norm'], daisy[dcol]['raw']
    diff = np.abs(sv - dv)
    r_norm = float(np.corrcoef(sv, dv)[0, 1])
    r_raw = float(np.corrcoef(sv, dr)[0, 1])
    sp = float(np.corrcoef(np.argsort(np.argsort(sv)),
                           np.argsort(np.argsort(dr)))[0, 1])
    print(f'\n  {s} vs DAISY {dcol}')
    print(f'    identical to DAISY min-max normalised: '
          f'{"YES" if diff.max() < 5e-3 else "NO"}  (max |diff| {diff.max():.3f}, '
          f'mean {diff.mean():.3f})')
    print(f'    Pearson r vs normalised {r_norm:+.4f}; vs raw {r_raw:+.4f}; '
          f'Spearman vs raw {sp:+.4f}')
    worst = np.argsort(-diff)[:5]
    print('    largest disagreements: ' + ', '.join(
        f'{AAS[k]} STEP {sv[k]:.2f} vs DAISY {dv[k]:.2f} (raw {dr[k]:g})'
        for k in worst))
    rep['step_vs_daisy'][s] = dict(max_abs_diff=float(diff.max()),
                                   mean_abs_diff=float(diff.mean()),
                                   pearson_vs_norm=r_norm,
                                   pearson_vs_raw=r_raw, spearman_vs_raw=sp)

print('\n' + sep)
print('  4. STEP vs PUBLISHED SCALES (peptides 0.5.0 + Biopython ProtParamData)')
print(sep)
LIB = {}
try:
    import peptides.tables as PT
    for fam in ('HYDROPHOBICITY',):
        for nm, tbl in getattr(PT, fam).items():
            if all(a in tbl for a in AAS):
                LIB[f'peptides/{fam}/{nm}'] = np.array([tbl[a] for a in AAS], float)
except Exception as e:                                   # pragma: no cover
    print('  peptides unavailable:', e)
try:
    from Bio.SeqUtils import ProtParamData as PPD
    for nm in ('kd', 'Flex', 'hw', 'em', 'ja', 'ms'):
        t = getattr(PPD, nm, None)
        if isinstance(t, dict) and all(a in t for a in AAS):
            LIB[f'biopython/{nm}'] = np.array([t[a] for a in AAS], float)
except Exception as e:                                   # pragma: no cover
    print('  biopython unavailable:', e)
try:
    import modlamp.core as MC
    import inspect as _i
    import re as _re
    _src = _i.getsource(MC.load_scale)
    for nm in _re.findall(r'^\s{8}"(\w+)": \{', _src, _re.M):
        try:
            _, tbl = MC.load_scale(nm)
        except Exception:
            continue
        if all(a in tbl and len(tbl[a]) == 1 for a in AAS):   # 1-D scales only
            LIB[f'modlamp/{nm}'] = np.array([tbl[a][0] for a in AAS], float)
except Exception as e:                                   # pragma: no cover
    print('  modlamp unavailable:', e)
print(f'  reference scales loaded: {len(LIB)}')

rep['best_matches'] = {}
for s, _ in pairs:
    sv = vec(STEP[s])
    scored = []
    for nm, rv in LIB.items():
        if rv.max() == rv.min():
            continue
        nv = minmax(rv)
        scored.append((float(np.abs(sv - nv).max()), nm,
                       float(np.corrcoef(sv, rv)[0, 1])))
    scored.sort()
    exact_raw = [nm for nm, rv in LIB.items() if np.abs(sv - rv).max() < 5e-3]
    exact_norm = [nm for mx, nm, _ in scored if mx < 5e-3]
    print(f'\n  {s}: EXACT match to a raw published scale: '
          f'{exact_raw or "none"}')
    print(f'  {s}: EXACT match after min-max normalisation: '
          f'{exact_norm or "none"}')
    print(f'  {s}: closest published scales (after min-max normalisation)')
    for mx, nm, r in scored[:5]:
        print(f'    max|diff| {mx:.3f}  r {r:+.4f}   {nm}')
    mx, nm, _ = scored[0]
    nv = minmax(LIB[nm])
    print(f'    per-residue, STEP vs {nm}:')
    print('      ' + '  '.join(f'{a}:{sv[k]:.2f}/{nv[k]:.2f}'
                               for k, a in enumerate(AAS))[:200])
    rep['best_matches'][s] = [dict(scale=nm, max_abs_diff=mx, pearson=r)
                              for mx, nm, r in scored[:5]]

print('\n' + sep)
print('  5. AREA_LOSS: static table or computed SASA?')
print(sep)
src = open(f'{ROOT}/data/compute_physchem.py').read()
print(f'  AREA_LOSS is a module-level dict literal of {len(STEP["AREA_LOSS"])} '
      f'constants, {len(set(STEP["AREA_LOSS"].values()))} distinct values.')
for pat in ('sasa', 'SASA', 'accessib', 'shrake', 'dssp', 'DSSP', 'freesasa',
            'naccess', 'surface', 'bound', 'unbound', 'complex'):
    print(f'    "{pat}" in data/compute_physchem.py: '
          f'{"YES" if pat in src else "no"}')
hits = subprocess.run(
    ['grep', '-rIl', '-e', 'freesasa', '-e', 'ShrakeRupley', '-e', 'DSSP',
     '-e', 'naccess', '--include=*.py', '.'],
    cwd=ROOT, capture_output=True, text=True).stdout.strip()
print(f'\n  repository-wide search for any SASA implementation '
      f'(freesasa / Bio.PDB.SASA.ShrakeRupley / DSSP / naccess):')
print('    ' + (hits.replace('\n', '\n    ') if hits else 'NO HITS'))
rep['area_loss'] = dict(n_values=20,
                        n_distinct=len(set(STEP['AREA_LOSS'].values())),
                        sasa_implementation_found=bool(hits))

print('\n' + sep)
print('  6. GLOBAL FEATURES (7)')
print(sep)
step_g = open(f'{ROOT}/models/dataset_physchem_only.py').read()
daisy_g = open(f'{ROOT}/baselines/repos/DAISY/DAISY.py').read()


def grab(text, start, end):
    i = text.index(start); j = text.index(end, i)
    return text[i:j]


a = re.findall(r'(\w+_peptide\.\w+\([^)]*\))', grab(
    step_g, 'def compute_global_features', 'global_diff = '))
b = re.findall(r'(\w+_peptide\.\w+\([^)]*\))', grab(
    daisy_g, 'def get_global_attributes_diff', 'global_diff_vector = '))
norm = lambda xs: [x.split('.', 1)[1] for x in xs]
print('  STEP  calls:', norm(a)[:7])
print('  DAISY calls:', norm(b)[:7])
print(f'\n  same 7 descriptors, same library, same order: '
      f'{norm(a)[:7] == norm(b)[:7]}')
print(f'  STEP combines them as tcr - epitope: '
      f'{"[t - e for t, e in zip(" in step_g}')
print(f'  DAISY combines them as tcr - epitope: '
      f'{"[tcr - pep for tcr, pep in zip(" in daisy_g}')
print(f'  STEP docstring: '
      f'{re.search(chr(34)*3 + r"(.*?)" + chr(34)*3, grab(step_g, "def compute_global_features", "hydrophobicity_table")).group(1)!r}')
rep['global_features'] = dict(step=norm(a)[:7], daisy=norm(b)[:7],
                              identical=norm(a)[:7] == norm(b)[:7])

json.dump(rep, open(f'{ROOT}/revision/evidence/r1_1a_r1_3_physchem.json', 'w'),
          indent=2, default=str)
print('\nsaved revision/evidence/r1_1a_r1_3_physchem.json')
