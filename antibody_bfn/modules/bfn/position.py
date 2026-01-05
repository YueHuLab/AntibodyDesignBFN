import torch
import torch.nn as nn

class PositionBFN(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def prior(self, shape, device):
        # Standard normal prior N(0, I)
        return torch.zeros(shape, device=device)

    def sample_theta(self, x_clean, t):
        """
        x_clean: (N, L, 3)
        t: (N,) in [0, 1]
        """
        # theta(t) is the mean of the distribution at time t.
        # In BFN, theta ~ N(alpha * x / (1 + alpha), alpha / (1 + alpha)^2 + ...)
        # Actually simplified:
        # We track the "input" parameter theta which accumulates information.
        # Let's use the 'parameters' space formulation.
        # Theta accumulates y. Theta ~ N(alpha * x, alpha)
        
        alpha_t = self.beta * t
        if isinstance(alpha_t, torch.Tensor):
            if alpha_t.dim() == x_clean.dim() - 1:
                alpha_t = alpha_t.unsqueeze(-1)
            elif alpha_t.dim() == 1:
                alpha_t = alpha_t.view(-1, 1, 1)
            else:
                alpha_t = alpha_t.unsqueeze(-1)
            
        mean = alpha_t * x_clean
        std = torch.sqrt(alpha_t)
        noise = torch.randn_like(x_clean)
        
        theta = mean + std * noise
        return theta

    def get_mean(self, theta, t):
        # Recover estimated mean from theta
        # If theta ~ N(alpha * x, alpha), then x_hat = theta / alpha
        # But we have prior N(0, I). 
        # Posterior mean = (theta + 0*1) / (alpha + 1)
        
        alpha_t = self.beta * t
        if isinstance(alpha_t, torch.Tensor):
            if alpha_t.dim() == theta.dim() - 1:
                alpha_t = alpha_t.unsqueeze(-1)
            elif alpha_t.dim() == 1:
                alpha_t = alpha_t.view(-1, 1, 1)
            else:
                alpha_t = alpha_t.unsqueeze(-1)
            
        return theta / (alpha_t + 1.0)
