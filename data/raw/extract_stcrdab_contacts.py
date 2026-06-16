import pandas as pd
import numpy as np
import requests
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import os
import pickle

df = pd.read_csv(os.path.expanduser('~/PhD_data/New_TCR/data/raw/stcrdab_human_mhci_filtered.tsv'), sep='\t')

results = []
output_dir = os.path.expanduser('~/PhD_data/New_TCR/data/raw/stcrdab_structures')
os.makedirs(output_dir, exist_ok=True)

parser = PDBParser(QUIET=True)

for idx, row in df.iterrows():
    pdb_id = row['pdb']
    pdb_file = f"{output_dir}/{pdb_id}.pdb"
    
    # Skip if already downloaded
    if not os.path.exists(pdb_file):
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            pdb_data = requests.get(pdb_url, timeout=30).text
            with open(pdb_file, 'w') as f:
                f.write(pdb_data)
        except Exception as e:
            print(f"Download error {pdb_id}: {e}")
            continue
    
    try:
        structure = parser.get_structure(pdb_id, pdb_file)
        
        tcr_chain = row['Bchain']
        epitope_chain = row['antigen_chain']
        
        # Get residues (protein only)
        tcr_residues = [r for r in structure[0][tcr_chain] if r.id[0] == ' ']
        epitope_residues = [r for r in structure[0][epitope_chain] if r.id[0] == ' ']
        
        # Convert 3-letter to 1-letter codes
        tcr_seq = ''.join([seq1(r.resname) for r in tcr_residues])
        epitope_seq = ''.join([seq1(r.resname) for r in epitope_residues])
        
        # Extract CDR3β region (residues 105-117 in IMGT numbering)
        # For now, use full chain - will extract CDR3 later with ANARCI
        
        # Distance matrix (CA atoms)
        dist_matrix = np.zeros((len(tcr_residues), len(epitope_residues)))
        for i, tcr_res in enumerate(tcr_residues):
            for j, epi_res in enumerate(epitope_residues):
                try:
                    dist = tcr_res['CA'] - epi_res['CA']
                    dist_matrix[i, j] = dist
                except:
                    dist_matrix[i, j] = np.nan
        
        results.append({
            'pdb': pdb_id,
            'tcr_chain': tcr_chain,
            'epitope_chain': epitope_chain,
            'tcr_full_seq': tcr_seq,
            'epitope_seq': epitope_seq,
            'resolution': row['resolution'],
            'distance_matrix': dist_matrix
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)}")
            
    except Exception as e:
        print(f"Error {pdb_id}: {e}")
        continue

output_file = os.path.expanduser('~/PhD_data/New_TCR/data/raw/stcrdab_contact_maps.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved {len(results)} structures to: {output_file}")
