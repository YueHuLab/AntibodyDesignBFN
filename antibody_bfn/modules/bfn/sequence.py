import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalBFN(nn.Module):
    def __init__(self, num_classes=20, num_steps=100, beta=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.beta = beta # Precision scalar

    def prior(self, shape, device):
        # Uniform prior on simplex implies uniform prior on parameters theta?
        # Usually theta initialized to 0 (uniform probabilities)
        return torch.zeros(shape + (self.num_classes,), device=device)

    def sample_sender(self, x_clean, t_step):
        """
        x_clean: (N, L) LongTensor or (N, L, K) One-hot FloatTensor
        t_step: scalar or (N,)
        
        Returns: y (Sender sample), (N, L, K)
        """
        if x_clean.dtype == torch.long:
            x_onehot = F.one_hot(x_clean, num_classes=self.num_classes).float()
        else:
            x_onehot = x_clean

        # Alpha(t) schedule - linear variance?
        # BFN paper uses specific schedule. Assuming beta * t for now or similar.
        # The prompt says "sample sender noise (y) ... alpha(t)".
        # Let's assume standard BFN continuous time formulation discretized.
        # sigma^2 = 1 / beta(t). 
        # Sender distribution y ~ N(beta * x_onehot, beta * I)
        
        # We need alpha(t) which accumulates precision.
        # alpha(t) = beta * t
        
        # If we just need to sample y for a step:
        # y = beta * x_onehot + sqrt(beta) * epsilon
        
        # But usually we update theta. 
        # theta(t) = theta_0 + y_accumulated.
        # y_accumulated ~ N(alpha(t) * x_onehot, alpha(t) * I)
        
        # Let's implement the 'sample_theta' directly from x_clean for training
        
        return self.sample_theta(x_onehot, t_step)

    def sample_theta(self, x_onehot, t):
        """
        Sample theta at time t given x_clean.
        theta(t) ~ N(alpha(t) * x_onehot, alpha(t) * I)
        """
        if x_onehot.dtype == torch.long:
            x_onehot = F.one_hot(x_onehot, num_classes=self.num_classes).float()

        # t is in [0, 1]
        alpha_t = self.beta * t
        
        if isinstance(alpha_t, torch.Tensor):
            # Assuming t is (N, L) or (N,), we want to broadcast to (N, L, K)
            # If t is (N, L), unsqueeze last -> (N, L, 1)
            # If t is (N,), view -> (N, 1, 1)
            if alpha_t.dim() == x_onehot.dim() - 1:
                alpha_t = alpha_t.unsqueeze(-1)
            elif alpha_t.dim() == 1:
                alpha_t = alpha_t.view(-1, 1, 1)
            else:
                 alpha_t = alpha_t.unsqueeze(-1) # Fallback
        
        mean = alpha_t * x_onehot
        std = torch.sqrt(alpha_t)
        noise = torch.randn_like(x_onehot)
        
        theta = mean + std * noise
        return theta

    def update(self, theta, y_step):
        # theta_new = theta + y_step
        # This update depends on how y is sampled.
        return theta + y_step

    def probabilities(self, theta):
        return F.softmax(theta, dim=-1)
