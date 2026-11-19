# -*- coding: utf-8 -*-
"""
Assemble the Zenodo staging folder for the revision deposit. Nothing is uploaded and
nothing in the repository is modified; files are hard-linked into the staging tree, so
the folder costs no extra disk and every staged file is the same inode as the original.
Run with --copy to take real copies instead.
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
OUT = Path('/home/muhammad-imran/Downloads/STEP_zenodo_v2_staging')
COPY = '--copy' in sys.argv

CKPT_DIRS = sorted(p.name for p in (BASE / 'models/checkpoints_v2').iterdir() if p.is_dir())
assert len(CKPT_DIRS) == 15, CKPT_DIRS

JOBS = []                                    # (source, destination relative to OUT)
for f in ('train.tsv', 'val.tsv', 'test.tsv', 'test_balanced.tsv'):
    JOBS.append((f'data/splits_600_epitope_v2/{f}', f'data/splits_600_epitope_v2/{f}'))
for f in ('train.tsv', 'val.tsv', 'test.tsv', 'test_balanced.tsv', 'split_report.json'):
    JOBS.append((f'data/splits_600_epitope_sim/{f}', f'data/splits_600_epitope_sim/{f}'))
for f in ('structure_priors_v2.npz', 'physchem_index_v2.json', 'physchem_maps_v2_extra.npy'):
    JOBS.append((f'data/embeddings/{f}', f'data/embeddings/{f}'))
for cd in CKPT_DIRS:
    for f in ('best_model.pt', 'history.json', 'results.json'):
        src = f'models/checkpoints_v2/{cd}/{f}'
        if (BASE / src).exists():
            JOBS.append((src, src))
for f in sorted(os.listdir(BASE / 'results/baseline_rerun_v2')):
    JOBS.append((f'results/baseline_rerun_v2/{f}', f'results/baseline_rerun_v2/{f}'))

# The similarity-disjoint arm: three STEP checkpoints, DAISY retrained on the same
# split, and the evaluation outputs -- without these the result reported in the
# manuscript cannot be reproduced from the deposit.
SIM_DIRS = sorted(p.name for p in (BASE / 'models/checkpoints_sim').iterdir() if p.is_dir())
assert len(SIM_DIRS) == 3, SIM_DIRS
for cd in SIM_DIRS:
    for f in ('best_model.pt', 'history.json', 'results.json'):
        src = f'models/checkpoints_sim/{cd}/{f}'
        if (BASE / src).exists():
            JOBS.append((src, src))
JOBS.append(('baselines/repos/DAISY/model/DAISY_sim.pt',
             'models/baselines/DAISY_sim.pt'))
for f in sorted(os.listdir(BASE / 'results/sim_split_eval')):
    JOBS.append((f'results/sim_split_eval/{f}', f'results/sim_split_eval/{f}'))
# the fifteen runs' test predictions and validation-fitted thresholds, behind
# Table 2 and Figure 4c,d
for f in sorted(os.listdir(BASE / 'results/matrix_v2')):
    JOBS.append((f'results/matrix_v2/{f}', f'results/matrix_v2/{f}'))

# every script written or changed during the revision, plus the two added since
code = subprocess.run(['git', 'diff', '--name-only', '5d9361d..HEAD'], cwd=BASE,
                      capture_output=True, text=True, check=True).stdout.split()
code = sorted({c for c in code if c.endswith(('.py', '.pml', '.sh'))}
              | {'revision/evidence/apply_si_format.py',
                 'revision/evidence/stage_zenodo_v2.py'})
for c in code:
    if (BASE / c).exists():
        JOBS.append((c, f'code/{c}'))

readme = (OUT / 'README.md').read_bytes() if (OUT / 'README.md').exists() else None
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
if readme is not None:
    (OUT / 'README.md').write_bytes(readme)
    JOBS.append((None, 'README.md'))
rows, total, linked, copied = [], 0, 0, 0
for src, dst in JOBS:
    t = OUT / dst
    t.parent.mkdir(parents=True, exist_ok=True)
    if src is not None:                       # README.md is already in place
        s = BASE / src
        try:
            if COPY:
                raise OSError
            os.link(s, t)
            linked += 1
        except OSError:
            shutil.copy2(s, t)
            copied += 1
    h = hashlib.sha256()
    with open(t, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    n = t.stat().st_size
    total += n
    rows.append((dst, n, h.hexdigest()))

commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=BASE,
                        capture_output=True, text=True, check=True).stdout.strip()
with open(OUT / 'MANIFEST.tsv', 'w') as fh:
    fh.write(f"# STEP revision deposit staging manifest\n")
    fh.write(f"# repository commit\t{commit}\n")
    fh.write(f"# files\t{len(rows)}\tbytes\t{total}\n")
    fh.write("path\tbytes\tsha256\n")
    for p, n, h in rows:
        fh.write(f"{p}\t{n}\t{h}\n")

groups = {}
for p, n, h in rows:
    key = p.split('/')[0] if not p.startswith(('data/', 'models/', 'results/')) \
        else '/'.join(p.split('/')[:2])
    groups.setdefault(key, [0, 0])
    groups[key][0] += 1
    groups[key][1] += n
print(f"staged at {OUT}")
print(f"commit {commit}")
print(f"{len(rows)} files, {total/1e6:.1f} MB ({linked} hard-linked, {copied} copied)\n")
for k in sorted(groups):
    c, n = groups[k]
    print(f"  {k:36s} {c:4d} files  {n/1e6:10.2f} MB")
