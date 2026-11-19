# Fig 5d: 4QOK interface with the corrected high-attention residues.
# Chains: A MHC heavy, B b2m, C epitope EAAGIGILTV (1-10),
#         D TCRa, E TCRb (CDR3b CAWSETGLGTGELFF at 91-105).
reinitialize
set ray_opaque_background, 1
bg_color white
load data/pdb_fig6/4QOK.pdb, cx
hide everything
remove solvent
remove not polymer

select mhc,     cx and chain A+B
select epitope, cx and chain C
select cdr3b,   cx and chain E and resi 91-105
select tcr_rest, cx and chain D+E and not cdr3b

show cartoon, mhc or epitope or cdr3b or tcr_rest
color grey80, tcr_rest
color lightpink, mhc
color magenta, epitope
color marine, cdr3b
set cartoon_transparency, 0.55, mhc
set cartoon_transparency, 0.70, tcr_rest

# the three top-attention positions on each side
select epi_hot,  epitope and resi 4+5+8
select cdr_hot,  cdr3b and resi 95+97+102
show sticks, (epi_hot or cdr_hot) and not (name C+N+O)
color orange, epi_hot and elem C
color deepteal, cdr_hot and elem C
show spheres, (epi_hot or cdr_hot) and name CA
set sphere_scale, 0.30, (epi_hot or cdr_hot) and name CA

distance contacts, epi_hot and name CA, cdr_hot and name CA, 8.0, 0
color grey40, contacts
hide labels, contacts
set dash_width, 2.0
set dash_gap, 0.35

set cartoon_fancy_helices, 1
set ray_shadows, 0
set antialias, 2
set specular, 0.2
set ambient, 0.28
set direct, 0.55

# residue labels, offset per residue so they do not collide
set label_size, 19
set label_color, black
set label_outline_color, white
set label_font_id, 7
label (epi_hot or cdr_hot) and name CA, "%s-%s" % (resn, resi)
set label_position, (-3.2,  2.6, 2.0), epitope and resi 4 and name CA
set label_position, ( 2.6,  1.0, 2.0), epitope and resi 5 and name CA
set label_position, (-3.4, -2.4, 2.0), epitope and resi 8 and name CA
set label_position, ( 3.0,  2.2, 2.0), cdr3b and resi 102 and name CA
set label_position, ( 3.4, -2.2, 2.0), cdr3b and resi 95 and name CA
set label_position, (-1.0, -3.4, 2.0), cdr3b and resi 97 and name CA

orient epitope or cdr3b
turn x, -12
zoom (epitope or cdr3b), 4
png revision/evidence/figures/Figure_5d_rebuilt_4QOK.png, width=2400, height=1900, dpi=600, ray=1
