"""
FIG6_FINAL_V6.py
Reference-matched rendering:
  - Wide view: cartoon faded to ~0.85 transparency, only peptide + CDR3 sticks visible solid
  - Zoom inset: NO cartoon, only atoms-and-bonds sticks for interacting residues
  - H-bond / contact dashes between close peptide and CDR3 atoms
  - Thin small labels, regular weight (no bold, no shadow boxes)
"""
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from Bio.PDB import PDBParser

PDB_DIR = Path('data/pdb_fig6')
CAM_DIR = Path('results/fig6_scorecam')
PNG_DIR = Path('results/fig6_pymol_v8'); PNG_DIR.mkdir(parents=True, exist_ok=True)

PDB_CONFIG = [
    {'epitope': 'NLVPMVATV', 'pdb': '3GSN', 'mhc': 'H', 'b2m': 'L',
     'peptide': 'P', 'tcra': 'A', 'tcrb': 'B', 'pep_offset': 0,
     'category': 'Clinical viral (CMV pp65)', 'flip_wide': False},
    {'epitope': 'AAGIGILTV', 'pdb': '4QOK', 'mhc': 'A', 'b2m': 'B',
     'peptide': 'C', 'tcra': 'D', 'tcrb': 'E', 'pep_offset': 1,
     'category': 'Cancer (MART-1 melanoma)', 'flip_wide': True},
]

THREE_TO_ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
                'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
                'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}


def find_cdr3_in_chain(pdb_path, chain_id, cdr3_seq):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure('x', str(pdb_path))
    for model in s:
        for chain in model:
            if chain.id != chain_id:
                continue
            residues = [r for r in chain if r.id[0] == ' ']
            seq = ''.join(THREE_TO_ONE.get(r.get_resname(), 'X') for r in residues)
            for term in [cdr3_seq, cdr3_seq[1:] if cdr3_seq.startswith('C') else 'C'+cdr3_seq]:
                idx = seq.find(term)
                if idx != -1:
                    matched = residues[idx:idx + len(term)]
                    return matched[0].id[1], matched[-1].id[1], matched
            sel = [r for r in residues if 95 <= r.id[1] <= 108]
            if sel:
                return sel[0].id[1], sel[-1].id[1], sel
        break
    return None, None, []


def get_peptide_residues(pdb_path, peptide_chain):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure('x', str(pdb_path))
    for model in s:
        for chain in model:
            if chain.id == peptide_chain:
                return [r for r in chain if r.id[0] == ' ']
        break
    return []


def write_pml(cfg, cam, cdr3_seq, view_type, out_pml, out_png):
    pdb_path = PDB_DIR / f"{cfg['pdb']}.pdb"
    pep_residues = get_peptide_residues(pdb_path, cfg['peptide'])
    cdr3_start, cdr3_end, cdr3_residues = find_cdr3_in_chain(pdb_path, cfg['tcrb'], cdr3_seq)

    pep_attn = cam.mean(axis=0)
    tcr_attn = cam.mean(axis=1)
    pep_offset = cfg['pep_offset']; epi_len = len(cfg['epitope'])

    pep_assignments = []
    for i in range(epi_len):
        si = i + pep_offset
        if si < len(pep_residues):
            pep_assignments.append((pep_residues[si].id[1], float(pep_attn[i]),
                                    pep_residues[si].get_resname()))
    cdr3_assignments = []
    if cdr3_residues:
        for j, r in enumerate(cdr3_residues):
            ti = int(round(j * (len(tcr_attn)-1) / max(1, len(cdr3_residues)-1)))
            ti = min(ti, len(tcr_attn)-1)
            cdr3_assignments.append((r.id[1], float(tcr_attn[ti]), r.get_resname()))

    top_pep = sorted(pep_assignments, key=lambda x: -x[1])[:3]
    top_cdr3 = sorted(cdr3_assignments, key=lambda x: -x[1])[:3]

    pml = []
    pml.append(f"load {pdb_path}, complex")
    pml.append("hide everything")
    if cfg['pdb'] == '6RSY':
        pml.append("remove chain F+G+H+I+J")

    pml.append("alter complex, b=0.5")
    for resnum, attn, _ in pep_assignments:
        pml.append(f"alter (chain {cfg['peptide']} and resi {resnum}), b={attn:.4f}")
    for resnum, attn, _ in cdr3_assignments:
        pml.append(f"alter (chain {cfg['tcrb']} and resi {resnum}), b={attn:.4f}")
    pml.append("rebuild")

    pml.append(f"color salmon, chain {cfg['mhc']}")
    pml.append(f"color lightpink, chain {cfg['b2m']}")
    if cfg.get('tcra'):
        pml.append(f"color skyblue, chain {cfg['tcra']}")
    pml.append(f"color lightblue, chain {cfg['tcrb']}")

    pml.append("bg_color white")
    pml.append("set ray_opaque_background, on")
    pml.append("set ray_shadows, 0")
    pml.append("set antialias, 2")

    # Orient: MHC up, TCR down (used for both wide and zoom)
    pml.append(f"pseudoatom mhc_center, chain {cfg['mhc']}")
    pml.append(f"pseudoatom tcr_center, chain {cfg['tcrb']}")
    pml.append("orient (mhc_center or tcr_center)")
    pml.append("rotate z, 90")
    if cfg['flip_wide']:
        pml.append("rotate x, 180")
    pml.append("delete mhc_center tcr_center")

    if view_type == 'wide':
        # ---- Wide: faded cartoon, prominent sticks on peptide + CDR3 ----
        pml.append("show cartoon, complex")
        pml.append("set cartoon_transparency, 0.75")  # faded
        # Sticks for peptide & CDR3 (full opacity)
        pml.append(f"show sticks, chain {cfg['peptide']}")
        if cdr3_start is not None:
            pml.append(f"show sticks, chain {cfg['tcrb']} and resi {cdr3_start}-{cdr3_end}")
        pml.append("hide sticks, elem H")
        # Color the sticks by attention
        pml.append(f"spectrum b, blue_white_red, chain {cfg['peptide']}, minimum=0, maximum=1")
        if cdr3_start is not None:
            pml.append(f"spectrum b, blue_white_red, (chain {cfg['tcrb']} and resi {cdr3_start}-{cdr3_end}), minimum=0, maximum=1")
        pml.append("set stick_radius, 0.18")
        pml.append("zoom complex, 6")
    else:
        # ---- Zoom: NO cartoon, only sticks of peptide + CDR3 ----
        pml.append(f"show sticks, chain {cfg['peptide']}")
        if cdr3_start is not None:
            pml.append(f"show sticks, chain {cfg['tcrb']} and resi {cdr3_start}-{cdr3_end}")
        pml.append("hide sticks, elem H")
        pml.append("set stick_radius, 0.20")
        pml.append("set stick_ball, 1")
        pml.append("set stick_ball_ratio, 1.5")

        # Color sticks by attention
        pml.append(f"spectrum b, blue_white_red, chain {cfg['peptide']}, minimum=0, maximum=1")
        if cdr3_start is not None:
            pml.append(f"spectrum b, blue_white_red, (chain {cfg['tcrb']} and resi {cdr3_start}-{cdr3_end}), minimum=0, maximum=1")

        # H-bond / contact dashes between peptide and CDR3 (within 4.0 Å)
        if cdr3_start is not None:
            pml.append(f"distance contacts, chain {cfg['peptide']}, "
                       f"(chain {cfg['tcrb']} and resi {cdr3_start}-{cdr3_end}), 4.0")
            pml.append("hide labels, contacts")  # hide the distance numbers
            pml.append("color gray60, contacts")
            pml.append("set dash_radius, 0.04")
            pml.append("set dash_gap, 0.3")
            pml.append("set dash_length, 0.3")

        # Reference-style: SMALL THIN labels, no bold, no shadow, no background
        for resnum, _, resname in top_pep:
            three = resname.upper()
            pml.append(f"label (chain {cfg['peptide']} and resi {resnum} and name CA), \"{three}-{resnum}\"")
        for resnum, _, resname in top_cdr3:
            three = resname.upper()
            pml.append(f"label (chain {cfg['tcrb']} and resi {resnum} and name CA), \"{three}-{resnum}\"")

        pml.append("set label_size, 14")
        pml.append("set label_font_id, 5")  # Arial regular (not bold)
        pml.append("set label_color, black")
        pml.append("set label_outline_color, white")
        pml.append("set label_position, (0, 2.5, 2.5)")  # offset off the sticks

        if cdr3_start is not None:
            pml.append(f"zoom (chain {cfg['peptide']} or (chain {cfg['tcrb']} and resi {cdr3_start}-{cdr3_end})), 2")
        else:
            pml.append(f"zoom chain {cfg['peptide']}, 4")

    pml.append(f"png {out_png}, width=1500, height=1500, dpi=200, ray=1")
    pml.append("quit")

    Path(out_pml).write_text('\n'.join(pml))


def run_pymol(pml_path):
    r = subprocess.run(['pymol', '-cq', str(pml_path)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  PyMOL stderr: {r.stderr[:500]}")
    return r.returncode == 0


# =============================================================================
# Render
# =============================================================================
metadata = pd.read_csv(CAM_DIR / 'metadata.csv')
metadata_by_ep = {row['epitope']: row for _, row in metadata.iterrows()}

print("=" * 60)
print("Rendering reference-matched (faded wide, sticks-only zoom)")
print("=" * 60)

for cfg in PDB_CONFIG:
    cam = np.load(CAM_DIR / f"{cfg['epitope']}_cam.npy")
    cdr3 = metadata_by_ep[cfg['epitope']]['cdr3']
    print(f"\n--- {cfg['epitope']} / {cfg['pdb']} ---")
    for view in ['wide', 'zoom']:
        pml = PNG_DIR / f"{cfg['epitope']}_{view}.pml"
        png = PNG_DIR / f"{cfg['epitope']}_{view}.png"
        write_pml(cfg, cam, cdr3, view, pml, png)
        print(f"  Rendering {view}...")
        run_pymol(pml)
        print(f"  Saved: {png}" if png.exists() else "  ❌ Failed")


# =============================================================================
# Assemble
# =============================================================================
print("\n" + "=" * 60)
print("Assembling Figure 6")
print("=" * 60)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11, 'axes.linewidth': 1.2,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

cmap_div = plt.get_cmap('RdBu_r')

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.18,
                       left=0.08, right=0.97, top=0.92, bottom=0.05,
                       width_ratios=[1.0, 1.7])

for row_idx, cfg in enumerate(PDB_CONFIG):
    ep = cfg['epitope']
    cam = np.load(CAM_DIR / f"{ep}_cam.npy")
    cdr3 = metadata_by_ep[ep]['cdr3']
    conf = metadata_by_ep[ep]['confidence']
    pdb_path = PDB_DIR / f"{cfg['pdb']}.pdb"
    pep_residues = get_peptide_residues(pdb_path, cfg['peptide'])
    cdr3_start, cdr3_end, cdr3_residues = find_cdr3_in_chain(pdb_path, cfg['tcrb'], cdr3)

    # Heatmap
    ax_h = fig.add_subplot(gs[row_idx, 0])
    cdr3_len = len(cdr3); epi_len = len(ep)
    cam_trim = cam[:cdr3_len, :epi_len]
    im = ax_h.imshow(cam_trim, aspect='auto', cmap=cmap_div, vmin=0, vmax=1,
                     interpolation='nearest')
    if cdr3_residues and len(cdr3_residues) >= cdr3_len:
        y_labels = [f"{cdr3_residues[j].get_resname().upper()}{cdr3_residues[j].id[1]}" for j in range(cdr3_len)]
    else:
        y_labels = list(cdr3)
    pep_offset = cfg['pep_offset']
    x_labels = []
    for i in range(epi_len):
        si = i + pep_offset
        if si < len(pep_residues):
            x_labels.append(f"{pep_residues[si].get_resname().upper()}{pep_residues[si].id[1]}")
        else:
            x_labels.append(ep[i])
    ax_h.set_xticks(range(epi_len))
    ax_h.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')
    ax_h.set_yticks(range(cdr3_len))
    ax_h.set_yticklabels(y_labels, fontsize=9)
    ax_h.set_xlabel('Epitope', fontsize=11)
    ax_h.set_ylabel('CDR3β', fontsize=11)
    cbar = plt.colorbar(im, ax=ax_h, fraction=0.045, pad=0.02)
    cbar.set_label('Attention', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    ax_h.set_title(f"{cfg['category']} | {ep} | P(bind)={conf:.3f}",
                   fontsize=10, pad=10, loc='left')

    # Right composite
    ax_r = fig.add_subplot(gs[row_idx, 1])
    ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1); ax_r.axis('off')
    wide_path = PNG_DIR / f"{ep}_wide.png"
    zoom_path = PNG_DIR / f"{ep}_zoom.png"
    if wide_path.exists():
        ax_r.imshow(plt.imread(wide_path), extent=[0.02, 0.36, 0.05, 0.95],
                    aspect='auto', zorder=1)
    if zoom_path.exists():
        ax_r.imshow(plt.imread(zoom_path), extent=[0.42, 0.99, 0.05, 0.95],
                    aspect='auto', zorder=2)
    ax_r.text(0.19, 0.97, 'MHC', fontsize=12, ha='center', fontweight='bold',
              color='#a64b3c', transform=ax_r.transAxes)
    ax_r.text(0.19, 0.03, 'TCR', fontsize=12, ha='center', fontweight='bold',
              color='#2e6e99', transform=ax_r.transAxes)

    rect_wide = FancyBboxPatch((0.13, 0.40), 0.13, 0.20, boxstyle="round,pad=0.005",
                               linewidth=1.3, edgecolor='#7d3fbf', facecolor='none',
                               linestyle='--', transform=ax_r.transAxes, zorder=3)
    ax_r.add_patch(rect_wide)
    rect_inset = FancyBboxPatch((0.42, 0.05), 0.57, 0.90, boxstyle="round,pad=0.01",
                                linewidth=1.8, edgecolor='#7d3fbf', facecolor='none',
                                linestyle='--', transform=ax_r.transAxes, zorder=3)
    ax_r.add_patch(rect_inset)
    ax_r.plot([0.26, 0.42], [0.60, 0.95], color='#7d3fbf', linestyle='--',
              linewidth=1.3, transform=ax_r.transAxes, zorder=3)
    ax_r.plot([0.26, 0.42], [0.40, 0.05], color='#7d3fbf', linestyle='--',
              linewidth=1.3, transform=ax_r.transAxes, zorder=3)

    ax_r.text(0.705, 0.985, 'Epitope', fontsize=13, ha='center', fontweight='bold',
              color='#a64b3c', transform=ax_r.transAxes,
              bbox=dict(facecolor='white', alpha=0.95, edgecolor='none', pad=3))
    ax_r.text(0.705, 0.02, 'CDR3β', fontsize=13, ha='center', fontweight='bold',
              color='#2e6e99', transform=ax_r.transAxes,
              bbox=dict(facecolor='white', alpha=0.95, edgecolor='none', pad=3))

fig.text(0.015, 0.95, 'a', fontsize=24, fontweight='bold', va='top', ha='left')
fig.text(0.40, 0.95, 'b', fontsize=24, fontweight='bold', va='top', ha='left')
fig.text(0.015, 0.49, 'c', fontsize=24, fontweight='bold', va='top', ha='left')
fig.text(0.40, 0.49, 'd', fontsize=24, fontweight='bold', va='top', ha='left')

out = Path('results/Figure_6_ScoreCAM_Structural.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
print(f"\nSaved: {out}")
print(f"Saved: {out.with_suffix('.pdf')}")
