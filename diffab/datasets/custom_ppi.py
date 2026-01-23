"""
Custom PPI Dataset for single-sample overfit testing.
Uses a simple directory of PDB files with two interacting chains.
"""
import os
import random
import logging
import torch
from Bio import PDB
from torch.utils.data import Dataset

from ..utils.protein import parsers
from ._base import register_dataset


INTERFACE_DISTANCE_THRESHOLD = 8.0


def detect_interface_residues(chain_a_coords, chain_b_coords, threshold=INTERFACE_DISTANCE_THRESHOLD):
    """Detect interface residues between two chains based on CA distance."""
    dist = torch.cdist(chain_a_coords, chain_b_coords)
    min_dist_a = dist.min(dim=1)[0]
    min_dist_b = dist.min(dim=0)[0]
    return min_dist_a < threshold, min_dist_b < threshold


def preprocess_ppi_structure(pdb_path, receptor_chain, ligand_chain):
    """
    Process a PDB file with two chains for PPI design.
    
    Args:
        pdb_path: path to PDB file
        receptor_chain: chain ID of receptor (context)
        ligand_chain: chain ID of ligand (to design)
    """
    parser = PDB.PDBParser(QUIET=True)
    
    try:
        pdb_id = os.path.basename(pdb_path).replace('.pdb', '')
        structure = parser.get_structure(pdb_id, pdb_path)
        model = structure[0]
        
        # Get list of chains
        chains = [c.id for c in model.get_chains()]
        
        if receptor_chain not in chains:
            logging.warning(f"Receptor chain {receptor_chain} not found in {pdb_path}")
            return None
        if ligand_chain not in chains:
            logging.warning(f"Ligand chain {ligand_chain} not found in {pdb_path}")
            return None
        
        # Parse chains
        receptor_data, receptor_seqmap = parsers.parse_biopython_structure(model[receptor_chain])
        ligand_data, ligand_seqmap = parsers.parse_biopython_structure(model[ligand_chain])
        
        if receptor_data is None or ligand_data is None:
            return None
        
        # Detect interface
        receptor_ca = receptor_data['pos_heavyatom'][:, 1]
        ligand_ca = ligand_data['pos_heavyatom'][:, 1]
        interface_r, interface_l = detect_interface_residues(receptor_ca, ligand_ca)
        
        if interface_r.sum() == 0 or interface_l.sum() == 0:
            logging.warning(f"No interface found in {pdb_path}")
            return None
        
        return {
            'id': pdb_id,
            'receptor': receptor_data,
            'receptor_seqmap': receptor_seqmap,
            'receptor_chain': receptor_chain,
            'ligand': ligand_data,
            'ligand_seqmap': ligand_seqmap,
            'ligand_chain': ligand_chain,
            'interface_receptor': interface_r,
            'interface_ligand': interface_l,
        }
        
    except Exception as e:
        logging.warning(f"Error processing {pdb_path}: {e}")
        return None


class CustomPPIDataset(Dataset):
    """
    Simple PPI dataset for overfit testing.
    Loads PDB files from a directory and treats them as protein complexes.
    """
    
    def __init__(
        self,
        structure_dir='./data/ppi_overfit',
        receptor_chain='A',
        ligand_chain='C',
        design_chain='ligand',
        transform=None,
    ):
        super().__init__()
        self.structure_dir = structure_dir
        self.receptor_chain = receptor_chain
        self.ligand_chain = ligand_chain
        self.design_chain = design_chain
        self.transform = transform
        
        # Find all PDB files
        self.pdb_files = [
            os.path.join(structure_dir, f) 
            for f in os.listdir(structure_dir) 
            if f.endswith('.pdb')
        ]
        
        if len(self.pdb_files) == 0:
            raise FileNotFoundError(f"No PDB files found in {structure_dir}")
        
        logging.info(f"CustomPPIDataset: {len(self.pdb_files)} PDB files")
    
    def __len__(self):
        return len(self.pdb_files)
    
    def __getitem__(self, index):
        pdb_path = self.pdb_files[index]
        
        # Parse structure
        data = preprocess_ppi_structure(pdb_path, self.receptor_chain, self.ligand_chain)
        
        if data is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Apply transform
        if self.transform is not None:
            try:
                data = self.transform(data)
            except Exception as e:
                logging.warning(f"Transform failed: {e}")
                return self.__getitem__(random.randint(0, len(self) - 1))
        
        if data is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        return data


@register_dataset('custom_ppi')
def get_custom_ppi_dataset(cfg, transform):
    return CustomPPIDataset(
        structure_dir=cfg.structure_dir,
        receptor_chain=cfg.get('receptor_chain', 'A'),
        ligand_chain=cfg.get('ligand_chain', 'C'),
        design_chain=cfg.get('design_chain', 'ligand'),
        transform=transform,
    )
