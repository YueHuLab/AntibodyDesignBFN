import torch

def kabsch_rotation(P, Q):
    """
    Calculate the optimal rotation matrix R such that R @ P \approx Q.
    Args:
        P: (N, 3) Source coordinates.
        Q: (N, 3) Target coordinates.
    Returns:
        R: (3, 3) Rotation matrix.
    """
    # Center the point clouds
    centroid_P = P.mean(dim=0, keepdim=True)
    centroid_Q = Q.mean(dim=0, keepdim=True)
    P_c = P - centroid_P
    Q_c = Q - centroid_Q

    # Compute covariance matrix
    H = P_c.transpose(0, 1) @ Q_c

    # SVD
    U, S, V = torch.svd(H)

    # Compute rotation
    d = torch.sign(torch.det(V @ U.t()))
    diag = torch.ones(3, device=P.device)
    diag[2] = d
    R = V @ torch.diag(diag) @ U.t()

    return R, centroid_P.view(3), centroid_Q.view(3)

def align_to_reference(pos_heavyatom, mask_heavyatom, fragment_type):
    """
    Align the current structure to a standard frame using Heavy Chain Framework CA atoms.
    Args:
        pos_heavyatom: (L, A, 3)
        mask_heavyatom: (L, A)
        fragment_type: (L,) Tensor indicating residue types (1 for Heavy, 2 for Light, etc.)
    """
    # 1. Select residues that are CA AND belong to the Heavy Chain (type 1)
    ca_mask = mask_heavyatom[:, 1].bool()
    heavy_mask = (fragment_type == 1) # constants.Fragment.Heavy is 1
    
    combined_mask = torch.logical_and(ca_mask, heavy_mask)
    valid_indices = torch.where(combined_mask)[0]
    
    # 2. Pick a stable subset (e.g., first 20 residues of the heavy chain)
    # These are usually the most conserved framework regions
    if len(valid_indices) < 20:
        # Fallback to whatever heavy chain residues we have
        subset_indices = valid_indices
    else:
        subset_indices = valid_indices[:20]
    
    if len(subset_indices) < 3:
        # Final fallback if structure is too small
        return pos_heavyatom, torch.eye(3).to(pos_heavyatom), torch.zeros(3).to(pos_heavyatom)

    P_ca = pos_heavyatom[subset_indices, 1] # (N, 3)
    
    # 3. Construct the Rigid Frame
    origin = P_ca[0]
    x_vec = P_ca[-1] - origin
    x_vec = x_vec / (torch.norm(x_vec) + 1e-6)
    
    mid_vec = P_ca[len(P_ca)//2] - origin
    z_vec = torch.cross(x_vec, mid_vec, dim=0)
    z_vec = z_vec / (torch.norm(z_vec) + 1e-6)
    
    y_vec = torch.cross(z_vec, x_vec, dim=0)
    
    R_inv = torch.stack([x_vec, y_vec, z_vec], dim=1) # (3, 3)
    
    # 4. Apply Transform
    pos_centered = pos_heavyatom - origin.view(1, 1, 3)
    pos_aligned = torch.matmul(pos_centered, R_inv)
    
    return pos_aligned, R_inv, origin
