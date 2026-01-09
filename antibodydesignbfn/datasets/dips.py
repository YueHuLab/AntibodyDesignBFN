"""
DIPS-Plus Dataset for Universal PPI Interface Design.

This dataset loads protein-protein interaction structures from DIPS-Plus
and identifies interface residues for sequence design.
"""
import os
import random
import logging
import pickle
import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from glob import glob

try:
    import dill
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

from ..utils.protein import constants
from ._base import register_dataset


# Interface detection parameters
INTERFACE_DISTANCE_THRESHOLD = 8.0  # Angstroms

# Amino acid mapping
AA_NAME_TO_INDEX = {
    'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
    'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
    'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
    'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
}


def detect_interface_residues(chain_a_coords, chain_b_coords, threshold=INTERFACE_DISTANCE_THRESHOLD):
    """Detect interface residues between two chains based on CA distance."""
    dist = torch.cdist(chain_a_coords, chain_b_coords)
    min_dist_a = dist.min(dim=1)[0]
    min_dist_b = dist.min(dim=0)[0]
    return min_dist_a < threshold, min_dist_b < threshold


def parse_dips_dill_file(dill_path):
    """
    Parse a DIPS .dill file containing atom3.pair.Pair object.
    
    Returns:
        dict with 'receptor', 'ligand', and interface information
    """
    if not HAS_DILL:
        logging.warning("dill package not installed. Run: pip install dill")
        return None
    
    try:
        pair = dill.load(open(dill_path, 'rb'))
        
        # Extract dataframes for both chains
        df0 = pair.df0  # First chain
        df1 = pair.df1  # Second chain
        
        # Get unique residues (group by residue number)
        def extract_chain_data(df):
            # Get CA atoms only
            ca_df = df[df['atom_name'] == 'CA'].copy()
            ca_df = ca_df.drop_duplicates(subset=['residue']).reset_index(drop=True)
            
            if len(ca_df) == 0:
                return None
            
            # Extract coordinates
            coords = torch.tensor(ca_df[['x', 'y', 'z']].values, dtype=torch.float32)
            
            # Extract sequence
            aa_indices = []
            for resname in ca_df['resname'].values:
                if resname in AA_NAME_TO_INDEX:
                    aa_indices.append(AA_NAME_TO_INDEX[resname])
                else:
                    aa_indices.append(20)  # Unknown
            aa = torch.tensor(aa_indices, dtype=torch.long)
            
            # Create chain_id and resseq
            n_res = len(aa)
            chain_id = [ca_df['chain'].iloc[0]] * n_res if 'chain' in ca_df.columns else ['A'] * n_res
            resseq = torch.tensor(ca_df['residue'].values.astype(int), dtype=torch.long)
            res_nb = torch.arange(n_res, dtype=torch.long)
            icode = [''] * n_res
            
            # Create placeholder heavy atom coords (14 atoms per residue)
            pos_heavyatom = torch.zeros(n_res, 14, 3, dtype=torch.float32)
            pos_heavyatom[:, 1, :] = coords  # CA position at index 1
            
            mask_heavyatom = torch.zeros(n_res, 14, dtype=torch.bool)
            mask_heavyatom[:, 1] = True  # Only CA is valid
            
            # Create torsion placeholders
            torsion = torch.zeros(n_res, 7, 2, dtype=torch.float32)
            mask_torsion = torch.zeros(n_res, 7, dtype=torch.bool)
            
            return {
                'aa': aa,
                'chain_id': chain_id,
                'resseq': resseq,
                'res_nb': res_nb,
                'icode': icode,
                'pos_heavyatom': pos_heavyatom,
                'mask_heavyatom': mask_heavyatom,
                'torsion': torsion,
                'mask_torsion': mask_torsion,
            }
        
        receptor_data = extract_chain_data(df0)
        ligand_data = extract_chain_data(df1)
        
        if receptor_data is None or ligand_data is None:
            return None
        
        # Detect interface
        receptor_ca = receptor_data['pos_heavyatom'][:, 1]
        ligand_ca = ligand_data['pos_heavyatom'][:, 1]
        interface_r, interface_l = detect_interface_residues(receptor_ca, ligand_ca)
        
        if interface_r.sum() == 0 or interface_l.sum() == 0:
            return None
        
        # Create ID from filename
        basename = os.path.basename(dill_path)
        pdb_id = basename.replace('.dill', '')
        
        return {
            'id': pdb_id,
            'pdb_id': pdb_id,
            'receptor': receptor_data,
            'receptor_seqmap': None,
            'receptor_chain': 'A',
            'ligand': ligand_data,
            'ligand_seqmap': None,
            'ligand_chain': 'B',
            'interface_receptor': interface_r,
            'interface_ligand': interface_l,
        }
        
    except Exception as e:
        logging.warning(f"Error parsing {dill_path}: {e}")
        return None


class DIPSDataset(Dataset):
    """
    Dataset for DIPS-Plus protein-protein interactions.
    
    Directory structure expected:
        data_dir/
            final/raw/dips/data/DIPS/filters/*.pdb
            
    Or flat structure:
        data_dir/*.pdb
    """
    
    MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB
    
    def __init__(
        self,
        data_dir='./data/dips_plus',
        processed_dir='./data/dips_processed',
        split='train',
        split_ratio=(0.9, 0.05, 0.05),  # train, val, test
        split_seed=2022,
        resolution_threshold=2.5,
        min_interface_size=5,
        max_length=500,
        transform=None,
        reset=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.split = split
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        self.resolution_threshold = resolution_threshold
        self.min_interface_size = min_interface_size
        self.max_length = max_length
        self.transform = transform
        
        os.makedirs(processed_dir, exist_ok=True)
        
        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)
        self._load_split()
        
        if self.db_conn is not None:
            self.db_conn.close()
            self.db_conn = None
    
    def _find_dill_files(self):
        """Find all .dill files in the data directory (DIPS format)."""
        patterns = [
            os.path.join(self.data_dir, '*.dill'),
            os.path.join(self.data_dir, 'raw', '**', '*.dill'),
            os.path.join(self.data_dir, '**', '*.dill'),
        ]
        
        dill_files = []
        for pattern in patterns:
            dill_files.extend(glob(pattern, recursive=True))
        
        dill_files = list(set(dill_files))  # Remove duplicates
        logging.info(f"Found {len(dill_files)} DIPS dill files in {self.data_dir}")
        return dill_files
    
    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'dips_structures.lmdb')
    
    def _load_structures(self, reset):
        if not os.path.exists(self._structure_cache_path) or reset:
            if os.path.exists(self._structure_cache_path):
                os.unlink(self._structure_cache_path)
            self._preprocess_structures()
        
        with open(self._structure_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)
        logging.info(f"Loaded {len(self.db_ids)} processed structures")
    
    def _preprocess_structures(self):
        """Preprocess all DIPS dill files and store in LMDB."""
        dill_files = self._find_dill_files()
        
        if len(dill_files) == 0:
            raise FileNotFoundError(
                f"No DIPS dill files found in {self.data_dir}. "
                "Please download DIPS-Plus from https://zenodo.org/record/5134732 "
                "and extract final_raw_dips.tar.gz"
            )
        
        logging.info(f"Preprocessing {len(dill_files)} DIPS files...")
        
        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for dill_path in tqdm(dill_files, desc='Preprocessing DIPS'):
                data = parse_dips_dill_file(dill_path)
                if data is None:
                    continue
                
                # Filter by interface size
                if (data['interface_receptor'].sum() < self.min_interface_size or
                    data['interface_ligand'].sum() < self.min_interface_size):
                    continue
                
                # Filter by length
                total_len = len(data['receptor']['aa']) + len(data['ligand']['aa'])
                if total_len > self.max_length:
                    continue
                
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))
        
        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)
        
        db_conn.close()
        logging.info(f"Preprocessed {len(ids)} valid structures")
    
    def _load_split(self):
        """Split data into train/val/test."""
        random.seed(self.split_seed)
        ids = self.db_ids.copy()
        random.shuffle(ids)
        
        n_total = len(ids)
        n_train = int(n_total * self.split_ratio[0])
        n_val = int(n_total * self.split_ratio[1])
        
        if self.split == 'train':
            self.ids_in_split = ids[:n_train]
        elif self.split == 'val':
            self.ids_in_split = ids[n_train:n_train + n_val]
        else:  # test
            self.ids_in_split = ids[n_train + n_val:]
        
        logging.info(f"Split '{self.split}': {len(self.ids_in_split)} samples")
    
    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
    
    def get_structure(self, id):
        self._connect_db()
        with self.db_conn.begin() as txn:
            return pickle.loads(txn.get(id.encode()))
    
    def __len__(self):
        return len(self.ids_in_split)
    
    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)
        
        if data is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        if self.transform is not None:
            try:
                data = self.transform(data)
            except Exception as e:
                logging.warning(f"Transform failed for {id}: {e}")
                return self.__getitem__(random.randint(0, len(self) - 1))
        
        if data is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        return data


@register_dataset('dips')
def get_dips_dataset(cfg, transform):
    return DIPSDataset(
        data_dir=cfg.data_dir,
        processed_dir=cfg.get('processed_dir', './data/dips_processed'),
        split=cfg.split,
        split_ratio=cfg.get('split_ratio', (0.9, 0.05, 0.05)),
        split_seed=cfg.get('split_seed', 2022),
        transform=transform,
    )
