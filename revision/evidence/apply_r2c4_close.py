"""Close [[PENDING: R2-C4]] from the rebuilt v2 CAM (revision/evidence/figures/cam_4qok_v2.json)."""
import sys
sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
PM, EM, EN = '±', '—', '–'
d = Docx(MS)
before = [d.ptext(p) for p in d.paragraphs()]

# ---------------------------------------------------------- Figure 5 caption
d.replace(
    f'c, Score-CAM activation map for the MART-1 epitope EAAGIGILTV (PDB 4QOK; '
    f'predicted binding probability 0.929 {PM} 0.122). d, Corresponding structural '
    'projection highlighting epitope residues (ALA-2, ALA-3, VAL-10) and '
    f'CDR3β residues (THR-96, GLY-97, LEU-98).',
    f'c, Attention map for the MART-1 epitope EAAGIGILTV (PDB 4QOK), averaged over '
    f'three training seeds; predicted binding probability 0.929 {PM} 0.122, with '
    f'individual seeds at 0.789, 1.000 and 0.999. Outlined cells are '
    f'Cα{EN}Cα contacts below 8 Å in 4QOK. d, Corresponding '
    'structural projection highlighting epitope residues (GLY-4, ILE-5, LEU-8) and '
    f'CDR3β residues (GLU-95, GLY-97, GLU-102). All three top-attention '
    f'epitope positions form a Cα{EN}Cα contact below 8 Å in the '
    'structure, as do two of the three CDR3β positions.',
    expected=1, label='R2-C4 Figure 5 caption, panels c and d')

# ---------------------------------------------------------- Results sentence
d.replace(
    'attention spanned epitope positions [[PENDING: R2-C4 ' + EM + ' ALA-2, ALA-3, '
    'VAL-10 if panel c keeps the CAM it currently plots; VAL-10, GLY-6, ILE-5 if it '
    'is rebuilt from the 4QOK CAM]] and CDR3β positions [[PENDING: R2-C4 '
    + EM + ' THR-96, GLY-97, GLU-95 as plotted; GLY-97, GLU-95, GLU-102 if rebuilt]] '
    '(Fig. 5c,d), highlighting the hydrophobic anchors and the central '
    'glycine' + EN + 'isoleucine motif characteristic of the MART-1 epitope family.',
    'attention concentrated on epitope positions ILE-5, GLY-4 and LEU-8 and on '
    f'CDR3β positions GLU-95, GLY-97 and GLU-102 (Fig. 5c,d) {EM} the central '
    'glycine' + EN + 'isoleucine motif characteristic of the MART-1 epitope family, '
    'together with its C-terminal hydrophobic anchor. All three top-attention '
    f'epitope positions form a Cα{EN}Cα contact below 8 Å in 4QOK, as '
    f'do two of the three CDR3β positions.',
    expected=1, label='R2-C4 Results, residue triplets')

# ---------------------------------------------------------- Methods 2.12 note
d.replace(
    'by mapping position-averaged attention scores onto residue B-factors and '
    'rendering with PyMOL51.',
    'by mapping position-averaged attention scores onto residue B-factors and '
    'rendering with PyMOL51. The maps reported here were recomputed on the '
    'retrained models and averaged over three seeds. The lower row of the submitted '
    'figure was drawn from an attention map computed for a different complex: the '
    'file it plotted is attributed to 5HHO in the attention-map metadata while being '
    'labelled with 4QOK residue numbers. The map rebuilt for 4QOK correlates with '
    'that file at r = 0.13, so the residue positions reported for that panel have '
    'changed.',
    expected=1, label='R2-C4 Methods 2.12, provenance note')

d.save()

d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
import difflib
for k, (b, a) in enumerate(zip(before, after)):
    if b != a:
        print(f'\n===== [{k}] =====')
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, b, a, autojunk=False).get_opcodes():
            if tag == 'equal':
                continue
            if b[i1:i2]:
                print(f'  - {b[i1:i2][:300]}')
            if a[j1:j2]:
                print(f'  + {a[j1:j2][:300]}')
import re
full = '\n'.join(after)
assert 'PENDING: R2-C4' not in full
print('\nmarkers left:', re.findall(r'\[\[[^\]]+\]\]', full))
import zipfile
with zipfile.ZipFile(MS) as z:
    assert z.testzip() is None
print('zip OK, paragraphs', len(after))
