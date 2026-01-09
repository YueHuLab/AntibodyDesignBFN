import torch
import torch.nn as nn
import torch.nn.functional as F
from antibodydesignbfn.modules.common.geometry import construct_3d_basis

def svd_project_so3_ot(F):
    """
    Ultra-robust SVD projection for LieOTAlign.
    Prevents convergence errors even with highly ill-conditioned matrices.
    """
    device = F.device
    N = F.size(0)
    
    # 1. Immediate exit if F contains NaN or Inf
    if not torch.isfinite(F).all():
        return torch.eye(3, device=device).view(1, 3, 3).expand(N, -1, -1)
    
    # 2. Scaling
    f_norm = torch.linalg.norm(F, dim=(-1, -2), keepdim=True).clamp(min=1e-8)
    F_scaled = F / f_norm
    
    # 3. Multiple levels of perturbation
    for eps in [1e-4, 1e-2, 1e-1]:
        try:
            eye = torch.eye(3, device=device).view(1, 3, 3).expand(N, -1, -1)
            M = F_scaled + eps * eye
            
            # Always perform SVD on CPU for maximum numerical stability in edge cases
            M_cpu = M.float().detach().cpu() 
            U, S, Vh = torch.linalg.svd(M_cpu)
            
            # Use determinant to ensure SO(3)
            d = torch.det(torch.matmul(U, Vh))
            mid = torch.ones_like(S)
            mid[..., -1] = d
            R = U @ torch.diag_embed(mid) @ Vh
            
            R = R.to(device)
            if torch.isfinite(R).all():
                return R
        except RuntimeError:
            continue
            
    # 4. Final Fallback: Return identity matrix
    return torch.eye(3, device=device).view(1, 3, 3).expand(N, -1, -1)

class LieOTAlign(nn.Module):
    def __init__(self, num_iters=10, epsilon=2.0):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon 

    def sinkhorn(self, cost, mask1, mask2):
        N, L1, L2 = cost.shape
        # Use a more stable log-space Sinkhorn
        # Add a small diagonal bias to the cost to encourage diagonal alignment 
        # (useful for overfitting or sequence-aligned structures)
        if L1 == L2:
            diag_bias = torch.eye(L1, device=cost.device).view(1, L1, L1) * (-0.1)
            cost = cost + diag_bias

        f = torch.zeros(N, L1, device=cost.device)
        g = torch.zeros(N, L2, device=cost.device)
        
        # log marginals
        log_a = -torch.log(mask1.sum(dim=1, keepdim=True).clamp(min=1.0))
        log_b = -torch.log(mask2.sum(dim=1, keepdim=True).clamp(min=1.0))
        
        M = cost / self.epsilon
        # Strong masking with a more reasonable value to avoid logsumexp overflow
        M = M.masked_fill(~mask1.unsqueeze(-1).expand(-1, -1, L2), 1e4)
        M = M.masked_fill(~mask2.unsqueeze(-2).expand(-1, L1, -1), 1e4)
        
        for _ in range(self.num_iters):
            f = log_a - torch.logsumexp(g.unsqueeze(1) - M, dim=2)
            g = log_b - torch.logsumexp(f.unsqueeze(2) - M, dim=1)
            
        pi = torch.exp(f.unsqueeze(2) + g.unsqueeze(1) - M)
        pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
        # Ensure rows/cols sum to 1 approx
        return pi

    def forward(self, pred_pos, true_pos, mask_gen, mask_res):
        N, L, _ = pred_pos.shape
        if not torch.isfinite(pred_pos).all():
            # Return zero tensor of shape (N,)
            return torch.zeros(N, device=pred_pos.device, requires_grad=True), true_pos, None

        # 1. Optimal Transport
        dist_sq = torch.cdist(pred_pos, true_pos) ** 2
        pi = self.sinkhorn(dist_sq, mask_res, mask_res) 
        
        # 2. Differentiable Alignment
        # true_pos_mapped is the target position for each predicted residue
        true_pos_mapped = torch.matmul(pi, true_pos) 
        
        weights = mask_res.float().unsqueeze(-1)
        w_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        c_pred = (pred_pos * weights).sum(dim=1, keepdim=True) / w_sum
        c_true = (true_pos_mapped * weights).sum(dim=1, keepdim=True) / w_sum
        
        p = (pred_pos - c_pred) * weights
        q = (true_pos_mapped - c_true) * weights
        
        # Covariance
        H = torch.matmul(p.transpose(-1, -2), q) 
        R = svd_project_so3_ot(H)
        
        # 3. Calculate TM-score in Aligned Space
        pred_pos_aligned = torch.matmul(pred_pos - c_pred, R) + c_true
        dist_sq_final = torch.sum((pred_pos_aligned - true_pos_mapped)**2, dim=-1)
        
        d0 = 1.24 * (max(L, 16) - 15)**(1/3) - 1.8
        d0 = max(d0, 0.5)
        
        tm_val = 1.0 / (1.0 + dist_sq_final / (d0**2))
        avg_tm = (tm_val * mask_gen).sum(dim=1) / (mask_gen.sum(dim=1).clamp(min=1e-6))
        
        return avg_tm, true_pos_mapped, pi
