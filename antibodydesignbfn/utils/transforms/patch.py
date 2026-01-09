import torch

from ._base import _mask_select_data, register_transform
from ..protein import constants
from ...modules.common.geometry import construct_3d_basis, global_to_local
from .align_utils import align_to_reference


@register_transform('patch_around_anchor')
class PatchAroundAnchor(object):

    def __init__(self, initial_patch_size=128, antigen_size=128):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.antigen_size = antigen_size

    def _align(self, data, anchor_idx):
        # Use the robust framework alignment targeting the heavy chain
        pos_aligned, R, t = align_to_reference(
            data['pos_heavyatom'], 
            data['mask_heavyatom'],
            data['fragment_type']
        )
        
        data['pos_heavyatom'] = pos_aligned
        # Apply Mask
        data['pos_heavyatom'] = data['pos_heavyatom'] * data['mask_heavyatom'][:, :, None]
        
        # Save transformation info
        data['patch_global_origin'] = t
        data['patch_global_rotation'] = R.t() # Store R such that P_global = P_local @ R_stored + t
        # align_to_reference returns P_local = (P_global - t) @ R_inv
        # So P_global = P_local @ R_inv.t() + t
        # Let's verify dimensions. align_utils returns R_inv (3,3).
        # We want to store the matrix that takes Local -> Global.
        # That is inv(R_inv) = R_inv.T
        
        return data

    def __call__(self, data):        
        anchor_flag = data['anchor_flag']   # (L,)
        anchor_points = data['pos_heavyatom'][anchor_flag, constants.BBHeavyAtom.CA]    # (n_anchors, 3)
        antigen_mask = (data['fragment_type'] == constants.Fragment.Antigen)
        antibody_mask = torch.logical_not(antigen_mask)

        if anchor_flag.sum().item() == 0:
            # Generating full antibody-Fv, no antigen given
            data_patch = _mask_select_data(
                data = data,
                mask = antibody_mask,
            )
            # Use the first residue as anchor if no explicit anchor
            # Mock anchor flag
            data_patch['anchor_flag'][0] = True 
            data_patch = self._align(data_patch, anchor_idx=0)
            return data_patch

        pos_alpha = data['pos_heavyatom'][:, constants.BBHeavyAtom.CA]  # (L, 3)
        dist_anchor = torch.cdist(pos_alpha, anchor_points).min(dim=1)[0]    # (L, )
        initial_patch_idx = torch.topk(
            dist_anchor,
            k = min(self.initial_patch_size, dist_anchor.size(0)),
            largest=False,
        )[1]   # (initial_patch_size, )

        dist_anchor_antigen = dist_anchor.masked_fill(
            mask = antibody_mask, # Fill antibody with +inf
            value = float('+inf')
        )   # (L, )
        antigen_patch_idx = torch.topk(
            dist_anchor_antigen, 
            k = min(self.antigen_size, antigen_mask.sum().item()), 
            largest=False, sorted=True
        )[1]    # (ag_size, )
        
        patch_mask = torch.logical_or(
            data['generate_flag'],
            data['anchor_flag'],
        )
        patch_mask[initial_patch_idx] = True
        patch_mask[antigen_patch_idx] = True

        patch_idx = torch.arange(0, patch_mask.shape[0])[patch_mask]

        data_patch = _mask_select_data(data, patch_mask)
        
        # Align using the first anchor residue found in the patch
        data_patch = self._align(data_patch, anchor_idx=None) # anchor_idx unused in _align new logic
        
        data_patch['patch_idx'] = patch_idx
        return data_patch
