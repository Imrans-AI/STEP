"""Supplementary Table S12: the five per-residue scales, in full."""
import ast
import copy
import sys

sys.path.insert(0, 'revision/evidence')
from lxml import etree

from docx_tools import Docx, W, WT

WTBL, WTR, WTC, WP = f'{{{W}}}tbl', f'{{{W}}}tr', f'{{{W}}}tc', f'{{{W}}}p'
WGRID, WGC = f'{{{W}}}tblGrid', f'{{{W}}}gridCol'
SI = 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'
BASE = '/home/muhammad-imran/PhD_data/TCR_Module_1/'
AAS = 'ACDEFGHIKLMNPQRSTVWY'
NAMES = ['HYDROPHOBICITY', 'CHARGE', 'FLEXIBILITY', 'REFRACTIVITY', 'AREA_LOSS']
_t = ast.parse(open(BASE + 'data/compute_physchem.py').read())
TAB = {n.targets[0].id: ast.literal_eval(n.value) for n in _t.body
       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
       and n.targets[0].id in NAMES}
HEAD = ['Residue', 'Hydrophobicity', 'Charge', 'Flexibility', 'Refractivity',
        'Fractional area loss']

d = Docx(SI)
body = d.tree.find(f'{{{W}}}body')
model = list(d.tree.iter(WTBL))[9]          # Table S9-ish: a simple 2-column table
mrow = list(model.iter(WTR))[0]
mcell = list(mrow.iter(WTC))[0]


def set_text(tc, s):
    ts = [t for p in tc.iter(WP) for t in p.iter(WT)]
    ts[0].text = s
    for t in ts[1:]:
        t.text = ''


tbl = copy.deepcopy(model)
for tr in list(tbl.iter(WTR)):
    tbl.remove(tr)
g = tbl.find(WGRID)
for gc in list(g):
    g.remove(gc)
for _ in HEAD:
    etree.SubElement(g, WGC).set(f'{{{W}}}w', '1340')

rows = [HEAD] + [[a] + [f'{TAB[n][a]:.2f}' for n in NAMES] for a in AAS]
for vals in rows:
    tr = copy.deepcopy(mrow)
    for tc in list(tr.iter(WTC)):
        tr.remove(tc)
    for v in vals:
        tc = copy.deepcopy(mcell)
        set_text(tc, v)
        tr.append(tc)
    tbl.append(tr)

# place it after the last table in the document, with a caption above
tables = list(d.tree.iter(WTBL))
last = tables[-1]
cap = copy.deepcopy([p for p in d.paragraphs()
                     if d.ptext(p).startswith('Table S10.')][0])
ts = [t for t in cap.iter(WT)]
ts[0].text = (
    'Table S12. The five per-residue physicochemical scales, as implemented. '
    'Values are the constants in data/compute_physchem.py and are applied as '
    'the pairwise absolute difference between a CDR3β residue and an '
    'epitope residue (Section 2.3.1). Hydrophobicity, charge, refractivity and '
    'fractional area loss lie on [0,1]; flexibility is used on its native range '
    'of 0.30–0.54.')
for t in ts[1:]:
    t.text = ''
last.addnext(tbl)
last.addnext(cap)
d.save()

d2 = Docx(SI)
t12 = list(d2.tree.iter(WTBL))[-1]
rr = [[' '.join(''.join(x.text or '' for x in c.iter(WT)).split())
       for c in tr.iter(WTC)] for tr in t12.iter(WTR)]
print(f'Table S12: {len(rr)} rows x {len(rr[0])} cols')
for r in rr[:4]:
    print('  | ' + ' | '.join(r))
print('  ...')
print('  | ' + ' | '.join(rr[-1]))
caps = [x for x in (d2.ptext(p) for p in d2.paragraphs()) if x.startswith('Table S')]
print('table captions:', [c[:12] for c in caps])
import zipfile
with zipfile.ZipFile(SI) as z:
    assert z.testzip() is None
print('zip OK')
