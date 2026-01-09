import torch
import torch.nn as nn
import torch.nn.functional as F
from antibodydesignbfn.modules.bfn.sequence import CategoricalBFN
from antibodydesignbfn.modules.bfn.position import PositionBFN
from antibodydesignbfn.modules.bfn.orientation import OrientationBFN
from antibodydesignbfn.modules.bfn.sidechain import SidechainBFN
from antibodydesignbfn.modules.bfn.receiver import AntibodyBFN_Receiver
from antibodydesignbfn.modules.common.geometry import repr_6d_to_rotation_matrix, compute_fape
from antibodydesignbfn.modules.common.ot_align import LieOTAlign

class AntibodyBFN_Core(nn.Module):
    def __init__(self, res_feat_dim, pair_feat_dim, num_steps, eps_net_opt={}, position_mean=[0.0, 0.0, 0.0], position_scale=[10.0], loss_weight={}, ot_opt={}, beta=1.0, schedule='linear'):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta  # Configurable BFN precision
        self.schedule = schedule  # 'linear' or 'cosine'
        self.num_classes = 22
        
        # Use only explicitly defined loss weights from config
        self.loss_weight = loss_weight.copy() if loss_weight else {}
        
        # Pass beta and schedule to flow components
        self.flow_seq = CategoricalBFN(num_classes=self.num_classes, num_steps=num_steps, beta=beta, schedule=schedule)
        self.flow_pos = PositionBFN()
        self.flow_ori = OrientationBFN()
        self.flow_ang = SidechainBFN()
        
        # Detect seq-only mode (no structure losses, except auxiliary angle loss)
        seq_only = not any(k in loss_weight and loss_weight.get(k, 0) > 0 for k in ['dist', 'fape', 'pos', 'rot'])
        self.receiver = AntibodyBFN_Receiver(res_feat_dim, pair_feat_dim, num_layers=6, encoder_opt=eps_net_opt, num_classes=self.num_classes, seq_only=seq_only)
        
        ot_iters = ot_opt.get('num_iters', 10)
        ot_eps = ot_opt.get('epsilon', 2.0)
        self.ot_align = LieOTAlign(num_iters=ot_iters, epsilon=ot_eps)
        
        self.register_buffer('_dummy', torch.empty([0, ]))
        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, batch):
        """
        Training step.
        """
        # Data
        x_seq = batch['aa'] # (N, L)
        x_pos = batch['pos_heavyatom'][:, :, 1] # CA (N, L, 3)
        x_pos = self._normalize_position(x_pos) # Normalize position
        
        # === Backbone atom coordinates (N, CA, C, O) ===
        # These are given directly to help model learn geometric features
        # Index: 0=N, 1=CA, 2=C, 3=O
        backbone_pos = batch['pos_heavyatom'][:, :, :4]  # (N, L, 4, 3)
        
        # Construct Rotation
        from antibodydesignbfn.modules.common.geometry import construct_3d_basis
        R_true = construct_3d_basis(
            batch['pos_heavyatom'][:, :, 1], # CA
            batch['pos_heavyatom'][:, :, 2], # C
            batch['pos_heavyatom'][:, :, 0], # N
        ) # (N, L, 3, 3)
        x_ori = R_true
        
        x_ang = batch['torsion'] # (N, L, 4) - sidechain chi angles
        mask_res = batch['mask']
        mask_gen = batch['generate_flag']
        
        # 1. Sample Time t
        N = x_seq.size(0)
        if 'fixed_t' in batch and batch['fixed_t'] is not None:
            t = torch.full((N,), batch['fixed_t'], device=x_seq.device)
        else:
            t = torch.rand(N, device=x_seq.device).clamp(1e-5, 1.0)
        
        # 2. Sample Theta (Parameters) from Clean Data
        t_expanded = t[:, None].expand(N, x_seq.size(1))
        
        theta_seq = self.flow_seq.sample_theta(x_seq, t_expanded)
        theta_pos = self.flow_pos.sample_theta(x_pos, t_expanded)
        theta_ori = self.flow_ori.sample_theta(x_ori, t_expanded)
        theta_ang_full = self.flow_ang.sample_theta(x_ang, t_expanded)
        theta_ang = self.flow_ang.get_angle(theta_ang_full)
        
        # 3. Receiver (Predict Clean Data)
        pair_feat = batch['pair_feat'] 
        
        # Convert theta to estimated mean/clean input for receiver
        # This is critical! theta scales with alpha(t), but receiver expects x_hat.
        inp_seq = self.flow_seq.probabilities(theta_seq)
        inp_pos = self.flow_pos.get_mean(theta_pos, t_expanded)
        inp_ori = self.flow_ori.get_rotation(theta_ori)
        inp_ang_raw = self.flow_ang.get_angle(theta_ang_full)
        
        # CRITICAL FIX: Mask CDR sidechain angles during training to prevent data leakage
        # Sidechain torsion angles reveal amino acid identity; model should learn backbone→sequence only
        mask_gen_ang = mask_gen.unsqueeze(-1)  # (N, L, 1)
        inp_ang = torch.where(mask_gen_ang, torch.zeros_like(inp_ang_raw), inp_ang_raw)

        # --- Recycling Pass (Disabled for Training to speed up) ---
        # We only do 1 pass during training. Inference will use recycling.
        prev_plddt, prev_iptm, prev_pae = None, None, None
        
        # Final prediction
        pred_seq, pred_pos, pred_ori_6d, pred_ang_sc, pred_plddt, pred_iptm, pred_pae = self.receiver(
            inp_seq, inp_pos, inp_ori, inp_ang, t, pair_feat, mask_res,
            backbone_pos=backbone_pos, prev_conf=prev_plddt, prev_iptm=prev_iptm, prev_pae=prev_pae
        )
        # --- End Recycling ---
        
        # 4. Training Losses
        losses = {}
        
        # BFN weighting
        weights = torch.full((N, 1), self.beta, device=x_seq.device)

        # === Sequence Loss ===
        x_seq_target = x_seq.clone()
        x_seq_target[x_seq >= 20] = -100
        loss_seq = F.cross_entropy(pred_seq.reshape(-1, self.num_classes), x_seq_target.flatten(), reduction='none', ignore_index=-100)
        loss_seq = loss_seq.reshape(N, -1)
        losses['seq'] = (loss_seq * mask_gen * weights).sum() / (mask_gen.sum() + 1e-8)

        # === Optional Structure Losses (only compute if in config) ===
        if any(k in self.loss_weight for k in ['dist', 'fape', 'ang']):
            pred_R = repr_6d_to_rotation_matrix(pred_ori_6d)
            pred_pos_unnorm = self._unnormalize_position(pred_pos)
            x_pos_unnorm = self._unnormalize_position(x_pos)
            
            if 'dist' in self.loss_weight and self.loss_weight['dist'] > 0:
                dist_pred = torch.cdist(pred_pos_unnorm, pred_pos_unnorm)
                dist_true = torch.cdist(x_pos_unnorm, x_pos_unnorm)
                mask_row = mask_gen.unsqueeze(-1).expand(N, mask_gen.size(1), mask_gen.size(1))
                mask_col = mask_gen.unsqueeze(1).expand(N, mask_gen.size(1), mask_gen.size(1))
                mask_pair = (mask_row | mask_col) & mask_res.unsqueeze(-1) & mask_res.unsqueeze(1)
                loss_dist = F.mse_loss(dist_pred, dist_true, reduction='none')
                losses['dist'] = (loss_dist * mask_pair).sum() / (mask_pair.sum() + 1e-8)
            
            if 'fape' in self.loss_weight and self.loss_weight['fape'] > 0:
                losses['fape'] = compute_fape(pred_R, pred_pos_unnorm, pred_pos_unnorm, x_ori, x_pos_unnorm, x_pos_unnorm, mask_gen, mask_gen)
            
            if 'ang' in self.loss_weight and self.loss_weight['ang'] > 0:
                s_hat, c_hat = pred_ang_sc[..., :4], pred_ang_sc[..., 4:]
                norm = torch.sqrt(s_hat**2 + c_hat**2 + 1e-8)
                s_hat, c_hat = s_hat / norm, c_hat / norm
                cos_diff = c_hat * torch.cos(x_ang) + s_hat * torch.sin(x_ang)
                loss_ang = 2 * (1 - cos_diff).sum(dim=-1)
                losses['ang'] = (loss_ang * mask_gen * weights).sum() / (mask_gen.sum() + 1e-8)

        # === Auxiliary Sidechain Angle Loss (for CDR regions) ===
        # This helps encoder learn geometric features from backbone
        # Input sidechain angles are masked (zero) for CDR, but we can still predict them
        if 'ang_aux' in self.loss_weight and self.loss_weight['ang_aux'] > 0:
            s_hat, c_hat = pred_ang_sc[..., :4], pred_ang_sc[..., 4:]
            norm = torch.sqrt(s_hat**2 + c_hat**2 + 1e-8)
            s_hat, c_hat = s_hat / norm, c_hat / norm
            cos_diff = c_hat * torch.cos(x_ang) + s_hat * torch.sin(x_ang)
            loss_ang_aux = 2 * (1 - cos_diff).sum(dim=-1)
            # Only compute on CDR (mask_gen) region
            losses['ang_aux'] = (loss_ang_aux * mask_gen).sum() / (mask_gen.sum() + 1e-8)

        losses['avg_t'] = t.mean()
        return losses

    @torch.no_grad()
    def sample(self, batch, sample_opt={}):
        N, L = batch['aa'].shape
        device = batch['aa'].device
        mask_gen = batch['generate_flag'].bool()
        pair_feat = batch['pair_feat'] 
        mask_res = batch['mask']
        deterministic = sample_opt.get('deterministic', False)  # Greedy/no noise sampling
        # Ground Truth Structure (always used in seq-only mode)
        x_seq = batch['aa']
        x_pos = batch['pos_heavyatom'][:, :, 1] # CA
        x_pos = self._normalize_position(x_pos)
        from antibodydesignbfn.modules.common.geometry import construct_3d_basis
        x_ori = construct_3d_basis(
            batch['pos_heavyatom'][:, :, 1],
            batch['pos_heavyatom'][:, :, 2],
            batch['pos_heavyatom'][:, :, 0],
        )
        x_ang = batch['torsion']
        
        # Backbone atom coordinates (N, CA, C, O) for explicit feature
        backbone_pos = batch['pos_heavyatom'][:, :, :4]  # (N, L, 4, 3)
        
        # Check if seq-only mode (no structure generation)
        seq_only = self.receiver.seq_only
        
        # Initialize Priors
        theta_seq = self.flow_seq.prior((N, L), device)
        
        if not seq_only:
            theta_pos = self.flow_pos.prior((N, L, 3), device)
            theta_ori = self.flow_ori.prior((N, L), device)
            theta_ang = self.flow_ang.prior((N, L), device)
        
        steps = self.num_steps
        delta_alpha = self.beta / steps
        sqrt_delta_alpha = torch.sqrt(torch.tensor(delta_alpha, device=device))
        
        for i in range(1, steps + 1):
            t = (i - 1) / steps
            t_tensor = torch.full((N,), t, device=device)
            t_expanded = t_tensor[:, None].expand(N, L)
            
            # For seq-only mode: use true backbone, but MASK CDR sidechain angles
            if seq_only:
                inp_pos = x_pos  # True backbone CA positions (expected for FixBB)
                inp_ori = x_ori  # True backbone orientations (expected for FixBB)
                # CRITICAL FIX: Mask CDR sidechain angles to prevent data leakage
                # Sidechain torsion angles reveal amino acid identity (e.g., Gly has no chi angles)
                mask_gen_ang = mask_gen.unsqueeze(-1)  # (N, L, 1)
                inp_ang = torch.where(mask_gen_ang, torch.zeros_like(x_ang), x_ang)
                
            else:
                # Sample context thetas
                theta_pos_ctx = self.flow_pos.sample_theta(x_pos, t_expanded)
                theta_ori_ctx = self.flow_ori.sample_theta(x_ori, t_expanded)
                theta_ang_ctx_full = self.flow_ang.sample_theta(x_ang, t_expanded)
                
                # Mix generated and context
                mask_gen_pos = mask_gen.unsqueeze(-1)
                theta_pos = torch.where(mask_gen_pos, theta_pos, theta_pos_ctx)
                theta_ori = torch.where(mask_gen.view(N, L, 1, 1), theta_ori, theta_ori_ctx)
                theta_ang = (
                    torch.where(mask_gen.unsqueeze(-1).unsqueeze(-1), theta_ang[0], theta_ang_ctx_full[0]),
                    torch.where(mask_gen.unsqueeze(-1).unsqueeze(-1), theta_ang[1], theta_ang_ctx_full[1]),
                    torch.where(mask_gen.unsqueeze(-1).unsqueeze(-1), theta_ang[2], theta_ang_ctx_full[2]),
                )
                
                inp_pos = self.flow_pos.get_mean(theta_pos, t_expanded)
                inp_ori = self.flow_ori.get_rotation(theta_ori)
                inp_ang = self.flow_ang.get_angle(theta_ang)
            
            # Sequence: always mix context
            theta_seq_ctx = self.flow_seq.sample_theta(x_seq, t_expanded)
            theta_seq = torch.where(mask_gen.unsqueeze(-1), theta_seq, theta_seq_ctx)
            inp_seq = self.flow_seq.probabilities(theta_seq)
            
            # Receiver forward
            res = self.receiver(inp_seq, inp_pos, inp_ori, inp_ang, t_tensor, pair_feat, mask_res,
                                backbone_pos=backbone_pos)
            pred_seq_logits, pred_pos, pred_ori_6d, pred_ang_sc, pred_plddt, pred_iptm, pred_pae = res
            
            # Update sequence - deterministic mode removes noise
            pred_seq_probs = F.softmax(pred_seq_logits, dim=-1)
            if deterministic:
                y_seq = delta_alpha * pred_seq_probs  # No noise
            else:
                y_seq = delta_alpha * pred_seq_probs + sqrt_delta_alpha * torch.randn_like(pred_seq_probs)
            theta_seq = theta_seq + y_seq
            
            # Update structure (only if not seq-only)
            if not seq_only:
                y_pos = delta_alpha * pred_pos + sqrt_delta_alpha * torch.randn_like(pred_pos)
                theta_pos = theta_pos + y_pos
                
                pred_R = repr_6d_to_rotation_matrix(pred_ori_6d)
                y_ori = delta_alpha * pred_R + sqrt_delta_alpha * torch.randn_like(pred_R)
                theta_ori = theta_ori + y_ori
                
                pred_ang = torch.atan2(pred_ang_sc[..., :4], pred_ang_sc[..., 4:])
                y_ang = delta_alpha * pred_ang + sqrt_delta_alpha * torch.randn_like(pred_ang)
                y_ang_expanded = y_ang.unsqueeze(-1).expand_as(theta_ang[1])
                theta_ang = (theta_ang[0], theta_ang[1] + y_ang_expanded, theta_ang[2] + delta_alpha)
        
        # Final outputs
        if theta_seq.size(-1) > 20:
            theta_seq[..., 20:] = -1e4
        final_seq = torch.argmax(theta_seq, dim=-1)
        
        if seq_only:
            final_R = x_ori
            final_pos = self._unnormalize_position(x_pos)
        else:
            final_R = self.flow_ori.get_rotation(theta_ori)
            final_pos = self._unnormalize_position(theta_pos / self.beta)
        
        from antibodydesignbfn.modules.common.so3 import rotation_to_so3vec
        v_0 = rotation_to_so3vec(final_R)
        
        return {
            0: (v_0, final_pos, final_seq),
            'pred_logits': pred_seq_logits,  # Model's direct prediction for PPL
            'plddt': pred_plddt,
            'iptm': pred_iptm,
            'pae': pred_pae
        }

    @torch.no_grad()
    def optimize(self, *args, **kwargs):
        return self.sample(*args, **kwargs)