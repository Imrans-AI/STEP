"""
R3-1a / R3-1b: transpose Table S10 and move the command strings to a footnote.

The table is 7 rows x 8 columns, and two of those columns (Training / inference
command, Provenance & reproduction status) carry strings long enough that the
table cannot fit a portrait page at a legible size, which is what R3-1a and
R3-1b object to. Transposing puts the six baselines across the page and the
eight fields down it, and the command strings move out of the grid entirely.

The ROC-AUC row is updated to the corrected values at the same time; leaving the
superseded column in a table being rebuilt would have reintroduced a mismatch.
"""
import copy
import sys

sys.path.insert(0, 'revision/evidence')
from lxml import etree

from docx_tools import Docx, W

WTBL, WTR, WTC, WP, WT = (f'{{{W}}}tbl', f'{{{W}}}tr', f'{{{W}}}tc',
                          f'{{{W}}}p', f'{{{W}}}t')
WGRID, WGC, WTCPR, WTCW = (f'{{{W}}}tblGrid', f'{{{W}}}gridCol',
                           f'{{{W}}}tcPr', f'{{{W}}}tcW')
SI = 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'
PM = '±'

d = Docx(SI)
body = d.tree.find(f'{{{W}}}body')
tbl = list(d.tree.iter(WTBL))[10]
rows = list(tbl.iter(WTR))
grid = [[c for c in tr.iter(WTC)] for tr in rows]
assert len(grid) == 7 and len(grid[0]) == 8, 'Table S10 shape changed'


def text_of(tc):
    return ' '.join(''.join(x.text or '' for x in tc.iter(WT)).split())


def set_text(tc, s):
    ts = [t for p in tc.iter(WP) for t in p.iter(WT)]
    assert ts
    ts[0].text = s
    for t in ts[1:]:
        t.text = ''


old = [[text_of(c) for c in r] for r in grid]
FIELD, BASE_ROWS = old[0], old[1:]
CMD_COL = FIELD.index('Training / inference command')
ROC_COL = [k for k, f in enumerate(FIELD) if f.startswith('ROC-AUC')][0]
ARCH_COL = FIELD.index('Architecture')

# corrected unseen-epitope values (NUMBERS_PASS.md §1)
ROC_NEW = {'STEP (this work)': f'0.747 {PM} 0.010', 'DAISY': '0.7013',
           'ERGO-LSTM': '0.5832', 'ERGO-AE': '0.6343', 'TEINet': '0.5239',
           'ATM-TCR': '0.5852'}
# CLAUDE.md: "structure-informed" is STEP's term alone; DAISY is not
ARCH_FIX = {'DAISY': 'Vision-guided multimodal, hierarchical physicochemical '
                     '(Condition-Adaptive Fusion); not structure-informed'}
FOOT = 'cdefgh'
cmds = []
for k, r in enumerate(BASE_ROWS):
    cmds.append((r[0], r[CMD_COL]))
    r[CMD_COL] = f'see note {FOOT[k]}'
    if r[0] in ROC_NEW:
        r[ROC_COL] = ROC_NEW[r[0]]
    if r[0] in ARCH_FIX:
        r[ARCH_COL] = ARCH_FIX[r[0]]

# ---- build the transposed grid: 8 field rows x 7 columns (label + 6 baselines)
new = [[FIELD[c]] + [BASE_ROWS[b][c] for b in range(6)] for c in range(8)]
new[0][0] = 'Field'

model_tr, model_tc = rows[0], grid[0][0]
tbl_children = list(tbl)
for tr in rows:
    tbl.remove(tr)
for ri, vals in enumerate(new):
    tr = copy.deepcopy(model_tr)
    for tc in list(tr.iter(WTC)):
        tr.remove(tc)
    for v in vals:
        tc = copy.deepcopy(model_tc)
        pr = tc.find(WTCPR)
        if pr is not None:
            w = pr.find(WTCW)
            if w is not None:
                w.set(f'{{{W}}}w', '1150' if vals is not new[0] else '1150')
        set_text(tc, v)
        tr.append(tc)
    tbl.append(tr)

g = tbl.find(WGRID)
for gc in list(g):
    g.remove(gc)
for _ in range(7):
    gc = etree.SubElement(g, WGC)
    gc.set(f'{{{W}}}w', '1150')

d.save()

# ---- footnote paragraph after the table, plus the caption note
d2 = Docx(SI)
t2 = [d2.ptext(p) for p in d2.paragraphs()]
cap = [i for i, x in enumerate(t2) if x.startswith('Table S10.')]
assert len(cap) == 1, cap
note = ('Commands, by baseline. ' + ' '.join(
    f'{FOOT[k]} {name}: {cmd}' for k, (name, cmd) in enumerate(cmds)))
ps = list(d2.paragraphs())
anchor = ps[cap[0]]
newp = copy.deepcopy(anchor)
for p_t in newp.iter(WT):
    p_t.text = ''
first = [t for t in newp.iter(WT)]
first[0].text = note
anchor.addnext(newp)
d2.save()

print('transposed Table S10 to 8 rows x 7 columns')
print('commands moved to a note of', len(note), 'characters')
d3 = Docx(SI)
t3 = list(d3.tree.iter(WTBL))[10]
for tr in t3.iter(WTR):
    print('  | ' + ' | '.join(
        ' '.join(''.join(x.text or '' for x in c.iter(WT)).split())[:26]
        for c in tr.iter(WTC)))
import zipfile
with zipfile.ZipFile(SI) as z:
    assert z.testzip() is None
print('zip OK')
