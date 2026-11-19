"""Fig 5a,b and Fig S2 text, on the structure-CDR3b rebuilds; plus the R1-4 Methods note."""
import sys
sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
SI = 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'
PM, EN = '±', '–'
d = Docx(MS)

d.replace(
    'For the CMV pp65 epitope NLVPMVATV (PDB 3GSN; P(bind) = 0.998 ' + PM + ' 0.004), '
    'attention concentrated on epitope positions ASN-1, VAL-3 and VAL-9 and '
    'CDR3β positions PRO-95, VAL-96 and THR-97 (Fig. 5a,b). These coincide with '
    'known contact hotspots in the RA14 TCR' + EN + 'NLVPMVATV structure, where '
    'CDR3β residues 95' + EN + '108 form van der Waals contacts with the peptide '
    'anchors.',
    f'For the CMV pp65 epitope NLVPMVATV (PDB 3GSN; P(bind) = 0.911 {PM} 0.140), '
    'attention concentrated on epitope positions MET-5, VAL-3 and ALA-7 and on '
    f'CDR3β positions THR-97, GLY-102 and GLY-99 (Fig. 5a,b). Two of the three '
    f'top epitope positions and two of the three CDR3β positions form a '
    f'Cα{EN}Cα contact below 8 Å in the structure.',
    expected=1, label='Results, 3GSN residues')

d.replace(
    'a, Gradient × input attribution map for the CMV pp65 epitope NLVPMVATV '
    f'(PDB 3GSN; predicted binding probability 0.998 {PM} 0.004), showing '
    'per-residue activation across CDR3β positions (rows) and epitope positions '
    '(columns). b, Activation projected onto the experimentally resolved '
    'TCR' + EN + 'pMHC structure; the magnified view highlights high-activation '
    'epitope residues (ASN-1, VAL-3, VAL-9) and CDR3β residues (PRO-95, VAL-96, '
    'THR-97) at the binding interface.',
    'a, Gradient × input attribution map for the CMV pp65 epitope NLVPMVATV '
    f'(PDB 3GSN; predicted binding probability 0.911 {PM} 0.140 over three training '
    f'seeds), with per-residue activation across CDR3β positions (rows) and '
    f'epitope positions (columns); outlined cells are Cα{EN}Cα contacts '
    'below 8 Å in the structure. b, Activation projected onto the '
    'experimentally resolved TCR' + EN + 'pMHC structure, highlighting epitope '
    'residues (VAL-3, MET-5, ALA-7) and CDR3β residues (THR-97, GLY-99, '
    'GLY-102) at the binding interface.',
    expected=1, label='Fig 5 caption, panels a and b')

# R1-4 + the CDR3b provenance note, in Methods 2.12
d.replace(
    'The map rebuilt for 4QOK correlates with that file at r = 0.13, so the residue '
    'positions reported for that panel have changed.',
    'The map rebuilt for 4QOK correlates with that file at r = 0.13, so the residue '
    'positions reported for that panel have changed. The same check was applied to '
    'the other three visualised complexes, and in each the CDR3β used by the '
    'submitted figure is not the CDR3β present in the structure it is drawn on '
    '(3GSN: ASSPVTGGIYGYT in the crystal against CASSKTQLWETQYF in the figure; '
    '6RSY: ASSLGFGRDVMR against ASSLGGNGYT; 1OGA: ASSSRSSYEQY against '
    'CASSIISQNEQFF). All attention maps reported here are computed on the '
    'CDR3β resolved in the corresponding structure, so attention and geometry '
    'describe the same complex. Finally, all four complexes used for visualisation '
    '(3GSN, 4QOK, 6RSY and 1OGA) are among the 198 structures from which the '
    'contact prior was derived, contributing five of those 198; the agreement '
    'between attention and observed contacts is therefore not independent of the '
    'prior, and should be read as internal consistency rather than external '
    'validation. Rebuilding the prior without them changes it by at most 0.011 in '
    'any cell (mean 0.001, r = 0.99999), so the prior itself is essentially '
    'unaffected by their inclusion.',
    expected=1, label='Methods 2.12, CDR3b provenance and R1-4')
d.save()

s = Docx(SI)
s.replace(
    'a, Gradient × input attribution map for the WT1 tumor neoantigen '
    'RMFPNAPYL (PDB 6RSY; predicted binding probability 0.775). b, Attribution map '
    'for the influenza M1 epitope GILGFVFTL (PDB 1OGA; predicted binding probability '
    '1.000), included as a well-characterized reference epitope. Activation is shown '
    'across CDR3β positions (rows) and epitope positions (columns).',
    'a, Gradient × input attribution map for the WT1 tumor neoantigen '
    f'RMFPNAPYL (PDB 6RSY; predicted binding probability 0.952 {PM} 0.073 over three '
    f'training seeds); top-attention positions are epitope PRO-7, PHE-3 and TYR-8 '
    f'and CDR3β ARG-105, PHE-99 and ARG-101. b, Attribution map for the '
    f'influenza M1 epitope GILGFVFTL (PDB 1OGA; 0.996 {PM} 0.006), included as a '
    'well-characterized reference epitope; top-attention positions are epitope '
    f'PHE-5, PHE-7 and ILE-2 and CDR3β SER-99, TYR-104 and ARG-98. Activation '
    f'is shown across CDR3β positions (rows) and epitope positions (columns), '
    f'each on the CDR3β resolved in the corresponding structure; outlined cells '
    f'are Cα{EN}Cα contacts below 8 Å.',
    expected=1, label='Fig S2 caption')
s.save()

for f in (MS, SI):
    dd = Docx(f)
    t = '\n'.join(dd.ptext(p) for p in dd.paragraphs())
    for bad in ('ASN-1, VAL-3 and VAL-9', 'PRO-95, VAL-96 and THR-97',
                'probability 0.775', 'probability 1.000)'):
        assert bad not in t, (f, bad)
print('text updated; superseded residue lists absent')
