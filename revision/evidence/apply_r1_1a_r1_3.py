"""
R1-1a / R1-3: physicochemical-scale provenance, applied to the manuscript.

Edits (all in revision/working/Manuscript_fv_1_REVISED.docx):
  2.3.1  - drop the false "Kyte-Doolittle scale" attribution on the hydrophobicity channel
         - name the flexibility channel for what it is
         - replace the "solvent-accessible surface area change upon binding" gloss on
           fractional area loss
         - correct "All channel values were min-max normalized to [0,1]", which is false
           for flexibility (native range 0.30-0.54)
         - new provenance paragraph: the DAISY relationship, and which values were changed
  2.3.5  - new sentence: the seven global descriptors are DAISY's, same library, same order
  3.x    - drop "binding-induced surface burial" from the SHAP paragraph
  4.x    - drop "buried surface area" from the Discussion

Citation numbers for the two new references are NOT inserted; they are left as
[[NEW REF]] markers for the author, since adding a reference renumbers the list.

Read-only on revision/as_submitted/.
"""
import copy
import sys

sys.path.insert(0, 'revision/evidence')
from lxml import etree

from docx_tools import Docx, W, WT

WR = f'{{{W}}}r'
WRPR = f'{{{W}}}rPr'
WVA = f'{{{W}}}vertAlign'
WP = f'{{{W}}}p'
WPPR = f'{{{W}}}pPr'
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'


def runs(p):
    return [r for r in p.iter(WR)]


def rtext(r):
    return ''.join(t.text or '' for t in r.iter(WT))


def is_sup(r):
    pr = r.find(WRPR)
    if pr is None:
        return False
    va = pr.find(WVA)
    return va is not None and va.get(f'{{{W}}}val') == 'superscript'


def make_run(model, text):
    r = etree.Element(WR, nsmap=model.nsmap)
    pr = model.find(WRPR)
    if pr is not None:
        r.append(copy.deepcopy(pr))
    t = etree.SubElement(r, WT)
    t.text = text
    if text != text.strip():
        t.set(XML_SPACE, 'preserve')
    return r


d = Docx(MS)
ps = list(d.paragraphs())
before = [d.ptext(p) for p in ps]
# Locate the two formatting models by content: paragraph indices move every
# time an earlier edit inserts a paragraph.
sup_model = next(r for p in ps for r in p.iter(WR)
                 if is_sup(r) and rtext(r) == '53,54')
bold_model = next(r for p in ps for r in p.iter(WR)
                  if rtext(r) == 'Threshold Selection.')

# ---------------------------------------------------- 2.3.1 channel list
# (1) hydrophobicity: the table is NOT Kyte-Doolittle (Spearman 0.91 against it,
#     tryptophan moves from rank 10 to rank 17), so the attribution is removed.
p43 = ps[43]
rs = runs(p43)
assert rtext(rs[0]) == 'Hydrophobicity (Kyte–Doolittle scale'
assert is_sup(rs[1]) and rtext(rs[1]) == '40' and rtext(rs[2]) == ')'
[t for t in rs[0].iter(WT)][0].text = 'Hydrophobicity'
for r in (rs[1], rs[2]):
    r.getparent().remove(r)

d.replace_in_paragraph(
    45, 'Flexibility (normalized B-factor)',
    'Flexibility (side-chain flexibility index)', label='R1-1a flexibility name')
d.replace_in_paragraph(
    47, 'Fractional area loss (solvent-accessible surface area change upon binding)',
    'Fractional area loss (a fixed per-residue scale, not a computed '
    'surface-area change)', label='R1-3 area-loss definition')

# (4) the normalisation claim. Run 0, then an inline [0,1] equation, then '.'.
p48 = ps[48]
r48 = runs(p48)
assert rtext(r48[0]) == 'All channel values were min-max normalized to '
assert rtext(r48[-1]) == '.'
[t for t in r48[0].iter(WT)][0].text = (
    'Hydrophobicity, charge, refractivity and fractional area loss are defined on ')
[t for t in r48[-1].iter(WT)][0].text = (
    '. Flexibility is used on its native range of 0.30–0.54, so that channel '
    'enters the network with a smaller dynamic range than the other four; no '
    'further normalization is applied after the pairwise difference is taken.')

# (5) provenance paragraph, immediately after the normalisation sentence
anchor = ps[49]
assert d.ptext(anchor).startswith('2.3.2 Structure-Informed Weighting')
body_model = runs(ps[48])[0]
newp = etree.Element(WP, nsmap=ps[48].nsmap)
newp.append(copy.deepcopy(ps[48].find(WPPR)))
PROV = [
    (0, 'Relationship to DAISY and provenance of the scales.'),
    (1, ' The five channels, the pairwise absolute-difference formulation above and '
        'the seven global descriptors of Section 2.3.5 follow the interaction-map '
        'representation introduced by DAISY'),
    (2, '28'),
    (1, '; the global descriptors are computed with the same library implementation, '
        'in the same order. STEP differs in the map dimensions (20 × 15 rather '
        'than 20 × 12) and in the structure-informed weighting of Section 2.3.2. '
        'The per-residue values, however, are specific to this work and are not '
        'DAISY’s. Charge reproduces DAISY’s formal-charge column rescaled to '
        '[0,1], except for histidine, which is given an intermediate value rather '
        'than the near-neutral one. Flexibility is the side-chain flexibility index '
        'of Bhaskaran and Ponnuswamy [[NEW REF 1]], used on its native range, with '
        'alanine at 0.35 rather than 0.36. The remaining three tables do not '
        'reproduce the published scales they are named after: hydrophobicity '
        'preserves the broad ordering of the Kyte–Doolittle scale'),
    (2, '40'),
    (1, ' (Spearman ρ = 0.91) but is not that scale, refractivity likewise departs '
        'from molar refractivity (ρ = 0.86, cysteine the largest discrepancy), and '
        'fractional area loss is not significantly rank-correlated with the published '
        'mean-fractional-area-loss scale (ρ = 0.40, P = 0.08). All five 20-value '
        'tables are listed in full in [[PENDING: R1-1a — new Supplementary table '
        'of the five per-residue scales]].'),
]
for kind, txt in PROV:
    newp.append(make_run({0: bold_model, 1: body_model, 2: sup_model}[kind], txt))
anchor.addprevious(newp)

# ---------------------------------------------------- 2.3.5 global features
ps = list(d.paragraphs())
auto = [i for i, p in enumerate(ps)
        if d.ptext(p) == 'Autocorrelation (Kyte–Doolittle)']
assert len(auto) == 1, auto
after = ps[auto[0] + 1]
assert d.ptext(after).startswith('The 7-dimensional feature vector'), d.ptext(after)

g = etree.Element(WP, nsmap=after.nsmap)
g.append(copy.deepcopy(after.find(WPPR)))
GLOB = [
    (1, 'These seven descriptors, their order and their formulation as a '
        'CDR3-minus-epitope difference are those of DAISY'),
    (2, '28'),
    (1, ', computed with the same implementation of the peptides package '
        '[[NEW REF 2]]; the autocorrelation uses the Kyte–Doolittle '
        'hydrophobicity table at lag 1.'),
]
for kind, txt in GLOB:
    g.append(make_run({1: runs(after)[0], 2: sup_model}[kind], txt))
after.addprevious(g)

# ---------------------------------------------------- burial wording
d.replace(
    'and the contributions of refractivity and area loss reflect the importance of '
    'residue volume and binding-induced surface burial in shaping interaction '
    'complementarity.',
    'and the contributions of refractivity and fractional area loss reflect the '
    'importance of residue volume and packing in shaping interaction '
    'complementarity. The low ranking of charge and flexibility should be read with '
    'care: the charge scale takes only four distinct values and the flexibility '
    'scale spans 0.30–0.54 against [0,1] for the other channels, so both enter '
    'the network with a smaller dynamic range (Section 2.3.1).',
    expected=1, label='R1-3 burial, Results')

d.replace('hydrophobicity, refractivity and buried surface area agree with',
          'hydrophobicity, refractivity and fractional area loss agree with',
          expected=1, label='R1-3 burial, Discussion')

d.save()

# ---------------------------------------------------------------- verify
d2 = Docx(MS)
after_t = [d2.ptext(p) for p in d2.paragraphs()]
print(f'paragraphs {len(before)} -> {len(after_t)} (expected +2)')
assert len(after_t) == len(before) + 2

import difflib
sm = difflib.SequenceMatcher(None, before, after_t, autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal':
        continue
    for k in range(i1, i2):
        print(f'\n[-{k}] {before[k][:700]}')
    for k in range(j1, j2):
        print(f'\n[+{k}] {after_t[k][:900]}')

for bad in ('Kyte–Doolittle scale', 'solvent-accessible surface area change',
            'binding-induced surface burial', 'buried surface area',
            'All channel values were min-max normalized'):
    assert bad not in '\n'.join(after_t), f'still present: {bad!r}'
print('\nall target strings removed')

import re
print('markers left:', re.findall(r'\[\[[^\]]+\]\]', '\n'.join(after_t)))

import zipfile
with zipfile.ZipFile(MS) as z:
    assert z.testzip() is None
print('zip OK')
