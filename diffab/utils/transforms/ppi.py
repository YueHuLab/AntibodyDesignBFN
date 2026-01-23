"""
Transforms for universal PPI (Protein-Protein Interaction) interface design.

These transforms work with DIPS-Plus data format:
- receptor: one protein chain
- ligand: interacting partner chain
- interface_receptor / interface_ligand: boolean masks
"""
import torch
import random
from ..protein import constants
from ._base import register_transform


@register_transform('merge_ppi_chains')
class MergePPIChains:
    """
    Merge receptor and ligand chains into a single data structure.
    Similar to MergeChains but for general PPI (not antibody-specific).
    """
    
    def __init__(self, design_chain='ligand'):
        """
        Args:
            design_chain: which chain to design ('ligand', 'receptor', or 'both')
        """
        super().__init__()
        self.design_chain = design_chain
    
    def _data_attr(self, data, name):
        if name in ('generate_flag', 'anchor_flag') and name not in data:
            return torch.zeros(data['aa'].shape, dtype=torch.bool)
        else:
            return data[name]
    
    def __call__(self, structure):
        receptor = structure['receptor']
        ligand = structure['ligand']
        interface_r = structure['interface_receptor']
        interface_l = structure['interface_ligand']
        
        if receptor is None or ligand is None:
            return None
        
        # Assign chain numbers
        receptor['chain_nb'] = torch.zeros_like(receptor['aa'])
        ligand['chain_nb'] = torch.ones_like(ligand['aa'])
        
        # Assign fragment type (using Antigen=2 for receptor, Heavy=0 for ligand)
        receptor['fragment_type'] = torch.full_like(receptor['aa'], fill_value=2)  # Context
        ligand['fragment_type'] = torch.full_like(ligand['aa'], fill_value=0)  # Design target
        
        # Initialize flags
        receptor['cdr_flag'] = torch.zeros_like(receptor['aa'])
        ligand['cdr_flag'] = torch.zeros_like(ligand['aa'])
        
        # Set generate_flag based on design_chain and interface
        receptor['generate_flag'] = torch.zeros_like(receptor['aa'], dtype=torch.bool)
        ligand['generate_flag'] = torch.zeros_like(ligand['aa'], dtype=torch.bool)
        
        if self.design_chain == 'ligand':
            ligand['generate_flag'] = interface_l
        elif self.design_chain == 'receptor':
            receptor['generate_flag'] = interface_r
        else:  # both
            receptor['generate_flag'] = interface_r
            ligand['generate_flag'] = interface_l
        
        # Set anchor (use non-interface residues of designed chain)
        receptor['anchor_flag'] = torch.zeros_like(receptor['aa'], dtype=torch.bool)
        ligand['anchor_flag'] = torch.zeros_like(ligand['aa'], dtype=torch.bool)
        
        # Anchor is the first interface residue of the designed chain
        if self.design_chain in ('ligand', 'both'):
            interface_indices = torch.where(interface_l)[0]
            if len(interface_indices) > 0:
                mid_idx = interface_indices[len(interface_indices) // 2]
                ligand['anchor_flag'][mid_idx] = True
        
        if self.design_chain in ('receptor', 'both'):
            interface_indices = torch.where(interface_r)[0]
            if len(interface_indices) > 0:
                mid_idx = interface_indices[len(interface_indices) // 2]
                receptor['anchor_flag'][mid_idx] = True
        
        # Merge properties
        data_list = [receptor, ligand]
        
        list_props = {
            'chain_id': [],
            'icode': [],
        }
        tensor_props = {
            'chain_nb': [],
            'resseq': [],
            'res_nb': [],
            'aa': [],
            'pos_heavyatom': [],
            'mask_heavyatom': [],
            'generate_flag': [],
            'cdr_flag': [],
            'anchor_flag': [],
            'fragment_type': [],
            'torsion': [],
            'mask_torsion': [],
        }
        
        for data in data_list:
            for k in list_props.keys():
                list_props[k].append(self._data_attr(data, k))
            for k in tensor_props.keys():
                tensor_props[k].append(self._data_attr(data, k))
        
        list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}
        
        data_out = {
            **list_props,
            **tensor_props,
            'id': structure['id'],
        }
        
        return data_out


@register_transform('mask_interface')
class MaskInterface:
    """
    Mask interface residues for design.
    Works with data that has 'interface_receptor' and 'interface_ligand' fields.
    """
    
    def __init__(self, design_chain='ligand', mask_ratio=1.0):
        """
        Args:
            design_chain: 'ligand', 'receptor', or 'both'
            mask_ratio: fraction of interface to mask (1.0 = all)
        """
        super().__init__()
        self.design_chain = design_chain
        self.mask_ratio = mask_ratio
    
    def __call__(self, structure):
        # This transform sets up the generate_flag for interface residues
        # The actual masking is done in merge_ppi_chains
        structure['design_chain'] = self.design_chain
        structure['mask_ratio'] = self.mask_ratio
        return structure


@register_transform('patch_around_interface')
class PatchAroundInterface:
    """
    Extract a patch of residues around the interface.
    Similar to PatchAroundAnchor but uses interface center.
    """
    
    def __init__(self, patch_size=128):
        super().__init__()
        self.patch_size = patch_size
    
    def __call__(self, data):
        if 'generate_flag' not in data:
            return data
        
        L = data['aa'].size(0)
        if L <= self.patch_size:
            # No need to patch
            return data
        
        # Find interface center
        gen_indices = torch.where(data['generate_flag'])[0]
        if len(gen_indices) == 0:
            return None
        
        center = gen_indices[len(gen_indices) // 2].item()
        
        # Expand around center
        half = self.patch_size // 2
        start = max(0, center - half)
        end = min(L, center + half)
        
        # Adjust if not enough
        if end - start < self.patch_size:
            if start == 0:
                end = min(L, self.patch_size)
            else:
                start = max(0, L - self.patch_size)
        
        # Select patch
        indices = torch.arange(start, end)
        
        patched = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.size(0) == L:
                patched[k] = v[indices]
            elif isinstance(v, list) and len(v) == L:
                patched[k] = [v[i] for i in range(start, end)]
            else:
                patched[k] = v
        
        return patched
