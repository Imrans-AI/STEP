"""Read-only: STEP's five tables vs DAISY's, exactly, plus cross-file identity."""
import ast, os, numpy as np
os.chdir('/home/muhammad-imran/PhD_data/TCR_Module_1')
AAS = 'ACDEFGHIKLMNPQRSTVWY'
NAMES = ['HYDROPHOBICITY', 'CHARGE', 'FLEXIBILITY', 'REFRACTIVITY', 'AREA_LOSS']

def tables(path):
    tree = ast.parse(open(path).read())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) \
           and n.targets[0].id in NAMES:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out

FILES = ['data/compute_physchem.py', 'data/BUILD_PHYSCHEM_v2.py',
         'COMPUTE_BIOLOGICAL_PHYSCHEM.py', 'COMPUTE_CANCER_PHYSCHEM.py',
         'TEST_STEP_UNIPMT_CORRECT.py']
base = tables(FILES[0])
print("=== cross-file identity of the five tables ===")
for f in FILES:
    t = tables(f)
    if not t:
        print(f"  {f:38} -> no literal tables found"); continue
    same = all(k in t and t[k] == base[k] for k in NAMES)
    missing = [k for k in NAMES if k not in t]
    diffs = {k: {a: (base[k][a], t[k][a]) for a in AAS if t[k][a] != base[k][a]}
             for k in NAMES if k in t and t[k] != base[k]}
    print(f"  {f:38} tables={len(t)} identical_to_compute_physchem={same}"
          f"{' missing='+str(missing) if missing else ''}"
          f"{' DIFFS='+str(diffs) if diffs else ''}")

def num(a):
    if isinstance(a, ast.UnaryOp):
        v = a.operand.value
        return -v if isinstance(a.op, ast.USub) else v
    return a.value

dt = ast.parse(open('baselines/repos/DAISY/DAISY.py').read())
D = None
for n in dt.body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'amino_acid_properties':
        D = {k.value: [num(x) for x in v.args[0].elts]
             for k, v in zip(n.value.keys, n.value.values)}
assert D and [D[a][2] for a in 'DEKRH'] == [-1.0, -1.0, 1.0, 1.0, 0.1]
COL = {'HYDROPHOBICITY': 0, 'REFRACTIVITY': 1, 'CHARGE': 2,
       'FLEXIBILITY': 3, 'AREA_LOSS': 4}

print("\n=== STEP vs DAISY: per-residue, raw and min-max normalised ===")
for k in NAMES:
    s = np.array([base[k][a] for a in AAS], float)
    r = np.array([D[a][COL[k]] for a in AAS], float)
    n = (r - r.min()) / (r.max() - r.min())
    eq_raw = int((np.abs(s - r) < 1e-9).sum())
    eq_norm = int((np.abs(s - n) < 5e-3).sum())
    print(f"\n  {k}  (DAISY column {COL[k]})")
    print(f"    residues equal to DAISY RAW: {eq_raw}/20 | "
          f"equal to DAISY MIN-MAX (<0.005): {eq_norm}/20")
    print(f"    {'aa':>3} {'STEP':>6} {'DAISYraw':>9} {'DAISYnorm':>10} {'STEP-norm':>10}")
    for i, a in enumerate(AAS):
        flag = '' if abs(s[i] - n[i]) < 5e-3 else '   <-- differs'
        print(f"    {a:>3} {s[i]:>6.2f} {r[i]:>9.3f} {n[i]:>10.3f} {s[i]-n[i]:>+10.3f}{flag}")
