# Fig 5b: 3GSN interface with the corrected high-attention residues.
reinitialize
set ray_opaque_background, 1
bg_color white
load data/pdb_fig6/3GSN.pdb, cx
hide everything
remove solvent
remove not polymer
select epitope, cx and chain P
select cdr3b,   cx and chain B and resi 92-104
select mhc,     cx and chain H+L
select tcr_rest, cx and chain A+B and not cdr3b
show cartoon, epitope or cdr3b or mhc or tcr_rest
color grey80, tcr_rest
color lightpink, mhc
color magenta, epitope
color marine, cdr3b
set cartoon_transparency, 0.55, mhc
set cartoon_transparency, 0.70, tcr_rest
select epi_hot, epitope and resi 3+5+7
select cdr_hot, cdr3b and resi 97+99+102
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
set label_size, 19
set label_color, black
set label_outline_color, white
set label_font_id, 7
label (epi_hot or cdr_hot) and name CA, "%s-%s" % (resn, resi)
set label_position, (-5.4,  3.6, 2.0), epitope and resi 3 and name CA
set label_position, ( 2.8,  2.2, 2.0), epitope and resi 5 and name CA
set label_position, (-3.2, -2.6, 2.0), epitope and resi 7 and name CA
set label_position, ( 3.2,  2.4, 2.0), cdr3b and resi 97 and name CA
set label_position, ( 3.4, -2.2, 2.0), cdr3b and resi 99 and name CA
set label_position, (-1.0, -3.4, 2.0), cdr3b and resi 102 and name CA
orient epitope or cdr3b
turn x, -12
zoom (epitope or cdr3b), 4
png revision/evidence/figures/Figure_5b_rebuilt_3GSN.png, width=2400, height=1900, dpi=600, ray=1
