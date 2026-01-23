import torch
import torch.nn as nn
import torch.nn.functional as F
from diffab.modules.encoders.ga import GAEncoder
from diffab.modules.bfn.orientation import svd_project_so3

class AntibodyBFN_Receiver(nn.Module):
    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}, num_classes=20, seq_only=False):
        super().__init__()
        self.res_feat_dim = res_feat_dim
        self.num_classes = num_classes
        self.seq_only = seq_only
        
        # Embeddings
        self.seq_embed = nn.Linear(num_classes, res_feat_dim)
        self.pos_embed = nn.Linear(3, res_feat_dim) # Embed mean position? 
        # Actually GAEncoder takes 't' directly. Maybe we just use pos directly.
        
        self.angle_embed = nn.Linear(4 * 2, res_feat_dim) # Concatenate sin/cos of 4 chi angles
        
        # Backbone atom coordinates embedding (N, CA, C, O relative to CA)
        # 4 atoms × 3 coords = 12 features
        self.backbone_embed = nn.Linear(12, res_feat_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim)
        )

        self.res_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 4, res_feat_dim), # seq + angle + time + backbone
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim)
        )

        # Shared Geometry-Aware Encoder
        # Ensure num_layers is not duplicated in encoder_opt
        opt = encoder_opt.copy()
        if 'num_layers' in opt:
            num_layers = opt.pop('num_layers')
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, ga_block_opt=opt)

        # Output Heads
        self.head_seq = nn.Linear(res_feat_dim, num_classes)
        self.head_pos = nn.Linear(res_feat_dim, 3)
        self.head_ori = nn.Linear(res_feat_dim, 6) # 6D representation for rotation
        self.head_ang = nn.Linear(res_feat_dim, 4 * 2) # sin/cos for 4 angles

        # Confidence Heads
        self.head_plddt = nn.Linear(res_feat_dim, 1)
        self.head_iptm = nn.Sequential(
            nn.Linear(res_feat_dim, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, 1)
        )
        self.head_pae = nn.Sequential(
            nn.Linear(res_feat_dim, res_feat_dim // 2),
            nn.ReLU(),
            nn.Linear(res_feat_dim // 2, 1)
        )
        
        # Feedback embedding: pLDDT, ipTM, and PAE back to features
        self.conf_embed = nn.Linear(1, res_feat_dim)
        self.iptm_embed = nn.Linear(1, res_feat_dim)
        self.pae_embed = nn.Linear(1, pair_feat_dim)
        
        # Project pair_feat to res_feat_dim for PAE head
        self.pair_proj = nn.Linear(pair_feat_dim, res_feat_dim)

    def forward(self, theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res, 
                backbone_pos=None, prev_conf=None, prev_iptm=None, prev_pae=None):
        N, L, _ = theta_seq.shape
        device = theta_seq.device
        
        # 1. Embeddings
        # Sequence: theta_seq is logits/accumulated evidence. Softmax to get probabilities.
        probs_seq = F.softmax(theta_seq, dim=-1) # (N, L, 20)
        emb_seq = self.seq_embed(probs_seq)      # (N, L, D)
        
        # Position: theta_pos is mean.
        # We pass this as 't' (translation) to GAEncoder.
        pos = theta_pos 
        
        # Orientation: theta_ori is F matrix. Project to R.
        # We pass this as 'R' to GAEncoder.
        rot = svd_project_so3(theta_ori)
        
        # Angle: theta_ang. Let's assume we extract expected angles.
        # For prototype, we assume theta_ang is (log_weights, means, precisions)
        # simplified: just use the means of the most likely component or weighted sum.
        # Let's use the 'get_angle' logic or simply pass the first moment if implemented.
        # Here we just take the means tensor directly if passed (simplified flow).
        # Assuming theta_ang passed here is already processed to (N, L, 4) or similar.
        # Wait, the core calls this. 
        # Let's assume theta_ang has shape (N, L, 4) representing estimated angles for embedding.
        # We embed as sin/cos.
        s_ang = torch.sin(theta_ang)
        c_ang = torch.cos(theta_ang)
        emb_ang = self.angle_embed(torch.cat([s_ang, c_ang], dim=-1)) # (N, L, D)
        
        # Time
        emb_t = self.time_embed(t.view(-1, 1, 1).expand(emb_seq.shape[:2] + (1,)))
        
        # Backbone coordinates (N, CA, C, O relative to CA)
        if backbone_pos is not None:
            # backbone_pos: (N, L, 4, 3) - positions of N, CA, C, O atoms
            ca_pos = backbone_pos[:, :, 1:2, :]  # (N, L, 1, 3) - CA position
            backbone_rel = backbone_pos - ca_pos  # (N, L, 4, 3) relative to CA
            backbone_flat = backbone_rel.reshape(N, L, -1)  # (N, L, 12)
            emb_backbone = self.backbone_embed(backbone_flat)  # (N, L, D)
        else:
            emb_backbone = torch.zeros(N, L, self.res_feat_dim, device=device)

        # Mix features
        res_feat = self.res_mixer(torch.cat([emb_seq, emb_ang, emb_t, emb_backbone], dim=-1))
        
        # Feedback recycling
        if prev_conf is not None:
            res_feat = res_feat + self.conf_embed(prev_conf.unsqueeze(-1))
        if prev_iptm is not None:
            res_feat = res_feat + self.iptm_embed(prev_iptm.view(-1, 1, 1))
        if prev_pae is not None:
            pair_feat = pair_feat + self.pae_embed(prev_pae.unsqueeze(-1))
        
        # 2. Encoder
        features = self.encoder(rot, pos, res_feat, pair_feat, mask_res)
        
        # 3. Heads
        pred_seq = self.head_seq(features)
        if pred_seq.size(-1) > 20:
            pred_seq[..., 20:] = -1e4

        # Seq-only mode: skip structure predictions (pos, ori) but keep angle prediction for aux loss
        if self.seq_only:
            pred_pos = pos  # Just return input
            pred_ori_6d = torch.zeros(N, L, 6, device=device)
            pred_ang_sc = self.head_ang(features)  # Still predict angles for auxiliary loss
            pred_plddt = torch.zeros(N, L, device=device)
            pred_iptm = torch.zeros(N, device=device)
            pred_pae = torch.zeros(N, L, L, device=device)
        else:
            # Full prediction
            pred_pos_local = self.head_pos(features)
            pred_pos = torch.matmul(rot, pred_pos_local.unsqueeze(-1)).squeeze(-1) + pos
            pred_ori_6d = self.head_ori(features)
            pred_ang_sc = self.head_ang(features)
            pred_plddt = torch.sigmoid(self.head_plddt(features)).squeeze(-1)
            
            masked_features = features * mask_res.unsqueeze(-1)
            global_features = masked_features.sum(dim=1) / (mask_res.sum(dim=1, keepdim=True) + 1e-8)
            pred_iptm = torch.sigmoid(self.head_iptm(global_features)).squeeze(-1)

            D = features.shape[-1]
            feat_row = features.view(N, L, 1, D).expand(N, L, L, D)
            feat_col = features.view(N, 1, L, D).expand(N, L, L, D)
            pair_combined = feat_row + feat_col + self.pair_proj(pair_feat)
            pred_pae = F.softplus(self.head_pae(pair_combined)).squeeze(-1) * 10.0
        return pred_seq, pred_pos, pred_ori_6d, pred_ang_sc, pred_plddt, pred_iptm, pred_pae
