import torch
import torch.nn as nn
from antibodydesignbfn.modules.common.geometry import normalize_vector

def svd_project_so3(F):
    """
    Project matrix F onto SO(3) using SVD.
    F = U S V^T
    R = U V^T * diag(1, 1, det(U V^T))
    """
    device = F.device
    # Add small noise to break degeneracy for SVD stability
    F = F + 1e-6 * torch.randn_like(F)
    
    if device.type == 'mps':
        F_cpu = F.float().cpu() # Ensure float32 on CPU
        U, S, Vh = torch.linalg.svd(F_cpu)
        
        # Calculate R on CPU to avoid MPS limitations with det/LU
        # R_temp = U @ Vh
        d = torch.det(torch.matmul(U, Vh))
        
        mid = torch.ones_like(S)
        mid[..., -1] = d
        
        R = U @ torch.diag_embed(mid) @ Vh
        return R.to(device)
    else:
        U, S, Vh = torch.linalg.svd(F)
        d = torch.det(U @ Vh)
        mid = torch.ones_like(S)
        mid[..., -1] = d
        R = U @ torch.diag_embed(mid) @ Vh
        return R

class OrientationBFN(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def prior(self, shape, device):
        return torch.zeros(shape + (3, 3), device=device)

    def sample_theta(self, R_clean, t):
        """
        R_clean: (N, L, 3, 3)
        t: (N,)
        """
        # F(t) accumulates noisy rotations.
        # Mean of F(t) is alpha(t) * R_clean.
        # Variance is proportional to alpha(t).
        
        alpha_t = self.beta * t
        if isinstance(alpha_t, torch.Tensor):
            if alpha_t.dim() == R_clean.dim() - 2:
                alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            elif alpha_t.dim() == 1:
                alpha_t = alpha_t.view(-1, 1, 1, 1)
            else:
                while alpha_t.dim() < R_clean.dim():
                    alpha_t = alpha_t.unsqueeze(-1)
            
        mean = alpha_t * R_clean
        std = torch.sqrt(alpha_t)
        noise = torch.randn_like(R_clean)
        
        F_t = mean + std * noise
        return F_t

    def get_rotation(self, F_t):
        return svd_project_so3(F_t)
