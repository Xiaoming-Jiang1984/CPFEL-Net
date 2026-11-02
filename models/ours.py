import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- helpers ----------
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class PatchSelector(nn.Module):
    def __init__(self, feat_dim, K=5000):
        super().__init__()
        self.K = K
        self.feat_dim = feat_dim

    def forward(self, fused):  # fused: [B, N, D]
        B, N, D = fused.shape
        K = min(self.K, N)
        norms = fused.norm(dim=-1)  # [B, N]
        topk_vals, topk_idx = torch.topk(norms, k=K, dim=1)
        rows = torch.arange(B, device=fused.device).unsqueeze(1).expand(-1, K)
        selected = fused[rows.reshape(-1), topk_idx.reshape(-1)].view(B, K, D)
        return selected, topk_idx

# ---------- main module ----------
class AdaptiveGatedAttention_one_size(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1, num_models=4, is_batch=False, select_K=5000):
        super().__init__()
        # hyperparams
        self.down_dim = in_dim
        self.ndim = in_dim
        self.is_batch = is_batch
        self.M = self.down_dim
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.num_models = num_models
        self.select_K = select_K
        self.selector = None

        # scoring params (kept frozen)
        self.score_alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.score_beta  = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        # === learnable threshold (STE) ===
        self.tau_logit = nn.Parameter(torch.tensor(0.0))
        self.ste_temperature = 0.5
        self.latest_gate_soft = 0.0

        # global model weights
        self.model_global_logits = nn.Parameter(0.01 * torch.randn(self.num_models))

        # transformer for disagreement
        trans_dim = self.down_dim
        self.trans_nhead = 4 if self.down_dim >= 64 else 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=self.trans_nhead,
            dim_feedforward=trans_dim * 2,
            dropout=0.4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.post_trans = nn.Sequential(
            nn.Linear(self.down_dim, self.down_dim),
            nn.ReLU()
        )

        # ABMIL
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)

        self.risk_predictor = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim)
        )

        initialize_weights(self)

    # ---------- scoring helpers ----------
    def feature_entropy(self, feat, eps=1e-8):
        a = feat.abs().clamp(min=1e-10)
        s = a.sum(dim=-1, keepdim=True) + eps
        p = a / s
        ent = -(p * (p + eps).log()).sum(dim=-1)
        return ent

    def compute_scores(self, reduced):
        # reduced: [B, S, N, D]
        B, S, N, D = reduced.shape
        flat = reduced.reshape(B * S * N, D)
        norms = flat.norm(dim=-1).clamp(max=1e4)
        ent = self.feature_entropy(flat)

        alpha = torch.sigmoid(self.score_alpha)
        beta  = torch.sigmoid(self.score_beta)
        total = alpha + beta + 1e-8
        alpha = alpha / total
        beta  = beta / total

        sc = alpha * norms + beta * ent
        sc = sc.view(B, S, N).permute(0, 2, 1)  # [B, N, S]

        sc_min, sc_max = sc.min(), sc.max()
        if (sc_max - sc_min) < 1e-6:
            sc = (sc - sc.mean()) / (sc.std() + 1e-8)
        else:
            sc = (sc - sc_min) / (sc_max - sc_min + 1e-8)
        return sc

    # ---------- forward ----------
    def forward(self, x, patch_batch_size=256, device=None):
        if device is None:
            device = x.device
        else:
            device = torch.device(device)
            x = x.to(device)

        _, S, _, _ = x.shape
        assert S == self.num_models

        # [B, N, S, D]
        reduced_pn = x.permute(0, 2, 1, 3)
        Bp, Np, Sp, Dp = reduced_pn.shape

        scores = self.compute_scores(x)          # [B, N, S]
        std_across_models = scores.std(dim=-1)   # [B, N]
        tau = torch.sigmoid(self.tau_logit)

        global_weights = F.softmax(
            self.model_global_logits.float(), dim=0
        ).view(1, 1, S, 1)

        fused_pn = torch.zeros(Bp, Np, Dp, device=device, dtype=reduced_pn.dtype)
        all_gate_soft = []

        for b in range(Bp):
            batch_patches = reduced_pn[b]
            std_b = std_across_models[b]

            for start in range(0, Np, patch_batch_size):
                end = min(start + patch_batch_size, Np)
                chunk = batch_patches[start:end]
                std_chunk = std_b[start:end]

                # ===== STE routing =====
                gate_soft = torch.sigmoid((tau - std_chunk) / self.ste_temperature)
                gate_hard = (gate_soft > 0.5).float()
                gate = gate_hard.detach() - gate_soft.detach() + gate_soft

                consensus_mask = gate.bool()
                disagreement_mask = ~consensus_mask
                all_gate_soft.append(gate_soft)

                if consensus_mask.any():
                    cons_idx = consensus_mask.nonzero(as_tuple=False).squeeze(1)
                    cons_feats = chunk[cons_idx]
                    w = global_weights.view(S).to(device=chunk.device, dtype=chunk.dtype)
                    fused = (cons_feats * w.view(1, S, 1)).sum(dim=1)
                    fused_pn[b, start:end][cons_idx] = fused

                if disagreement_mask.any():
                    dis_idx = disagreement_mask.nonzero(as_tuple=False).squeeze(1)
                    dis_feats = chunk[dis_idx]
                    trans_out = self.transformer(dis_feats)
                    fused = self.post_trans(trans_out.mean(dim=1))
                    fused_pn[b, start:end][dis_idx] = fused

        if self.training:
            self.latest_gate_soft = torch.cat(all_gate_soft, dim=0)

        if self.selector is not None:
            fused_pn, _ = self.selector(fused_pn)

        if self.is_batch:
            return self._forward_patch_batch(fused_pn, patch_batch_size, device=device)
        else:
            return self._forward_single_batch(fused_pn, device=device)

    # ---------- ABMIL helpers ----------
    def _forward_single_batch(self, H, device=None):
        B, N, D = H.shape
        risk_scores = []
        attn_list = []
        Z_list = []

        for i in range(B):
            Hi = H[i]
            V = self.attention_V(Hi)
            U = self.attention_U(Hi)
            logits = self.attention_w(V * U)
            A = F.softmax(logits.t(), dim=1)
            Z = A @ Hi
            risk = self.risk_predictor(Z).view(-1)
            attn_list.append(A)
            Z_list.append(Z)
            risk_scores.append(risk)

        return (attn_list, Z_list), torch.stack(risk_scores)

    def _forward_patch_batch(self, x, patch_batch_size=256, device=None):
        B, N, D = x.shape
        device = x.device
        attn_list, Z_list, risk_list = [], [], []

        for b in range(B):
            xb = x[b]
            sum_expA, weighted_sum = None, None
            logits_parts = []

            for start in range(0, N, patch_batch_size):
                end = min(start + patch_batch_size, N)
                H = xb[start:end]
                V = self.attention_V(H)
                U = self.attention_U(H)
                logits = self.attention_w(V * U)
                logits_parts.append(logits.detach().cpu())
                max_val = logits.max(dim=0, keepdim=True)[0]
                exp_part = torch.exp(logits - max_val)
                sum_exp_part = exp_part.sum(dim=0)
                weighted_part = (exp_part.unsqueeze(-1) * H.unsqueeze(1)).sum(dim=0)

                if sum_expA is None:
                    sum_expA = sum_exp_part
                    weighted_sum = weighted_part
                else:
                    sum_expA += sum_exp_part
                    weighted_sum += weighted_part

            Z = weighted_sum / (sum_expA.unsqueeze(-1) + 1e-8)
            risk = self.risk_predictor(Z.view(1, -1)).view(-1)
            logits_all = torch.cat(logits_parts, dim=0).t().to(device)
            attn_all = F.softmax(logits_all, dim=1)

            attn_list.append(attn_all)
            Z_list.append(Z)
            risk_list.append(risk)

        return (attn_list, Z_list), torch.stack(risk_list)



# Ablation-Global
class AdaptiveGatedAttention_GlobalOnly(nn.Module):
    """
    Ablation-1:
    All patches use global weighted fusion across models
    """
    def __init__(self, in_dim=1024, out_dim=1, num_models=4, is_batch=False, select_K=5000):
        super().__init__()

        self.down_dim = in_dim
        self.ndim = in_dim
        self.is_batch = is_batch
        self.M = self.down_dim
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.num_models = num_models
        self.select_K = select_K
        self.selector = None

        # global model weights
        self.model_global_logits = nn.Parameter(0.01 * torch.randn(self.num_models))

        # ABMIL
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)

        self.risk_predictor = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim)
        )

        initialize_weights(self)

    def forward(self, x, patch_batch_size=256, device=None):
        """
        x: [B, S, N, D]
        """
        if device is not None:
            x = x.to(device)

        B, S, N, D = x.shape
        assert S == self.num_models

        # [B, N, S, D]
        x = x.permute(0, 2, 1, 3)

        global_weights = F.softmax(self.model_global_logits, dim=0).view(1, 1, S, 1)

        fused = (x * global_weights).sum(dim=2)  # [B, N, D]

        if self.selector is not None:
            fused, _ = self.selector(fused)

        if self.is_batch:
            return self._forward_patch_batch(fused, patch_batch_size)
        else:
            return self._forward_single_batch(fused)

    # ===== ABMIL 部分完全复用 =====
    def _forward_single_batch(self, H):
        B, N, D = H.shape
        risk_scores, attn_list, Z_list = [], [], []

        for i in range(B):
            Hi = H[i]
            V = self.attention_V(Hi)
            U = self.attention_U(Hi)
            A = F.softmax(self.attention_w(V * U).t(), dim=1)
            Z = A @ Hi
            risk = self.risk_predictor(Z).view(-1)
            attn_list.append(A)
            Z_list.append(Z)
            risk_scores.append(risk)

        return (attn_list, Z_list), torch.stack(risk_scores)

    def _forward_patch_batch(self, x, patch_batch_size=256):
        B, N, D = x.shape
        attn_list, Z_list, risk_list = [], [], []

        for b in range(B):
            xb = x[b]
            logits_all, weighted_sum, sum_expA = [], None, None

            for start in range(0, N, patch_batch_size):
                xp = xb[start:start+patch_batch_size]
                V = self.attention_V(xp)
                U = self.attention_U(xp)
                logits = self.attention_w(V * U)
                logits_all.append(logits.detach().cpu())

                maxv = logits.max()
                expA = torch.exp(logits - maxv)
                sum_exp = expA.sum(dim=0)
                weighted = (expA.unsqueeze(-1) * xp.unsqueeze(1)).sum(dim=0)

                sum_expA = sum_exp if sum_expA is None else sum_expA + sum_exp
                weighted_sum = weighted if weighted_sum is None else weighted_sum + weighted

            Z = weighted_sum / (sum_expA.unsqueeze(-1) + 1e-8)
            risk = self.risk_predictor(Z.view(1, -1)).view(-1)

            A = F.softmax(torch.cat(logits_all, dim=0).t().to(x.device), dim=1)

            attn_list.append(A)
            Z_list.append(Z)
            risk_list.append(risk)

        return (attn_list, Z_list), torch.stack(risk_list)

# Ablation-Transformer
class AdaptiveGatedAttention_TransformerOnly(nn.Module):
    """
    Ablation-2:
    All patches use Transformer-based fusion across models
    """
    def __init__(self, in_dim=1024, out_dim=1, num_models=4, is_batch=False, select_K=5000):
        super().__init__()

        self.down_dim = in_dim
        self.ndim = in_dim
        self.is_batch = is_batch
        self.M = self.down_dim
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.num_models = num_models
        self.select_K = select_K
        self.selector = None

        # Transformer over model dimension
        trans_dim = self.down_dim
        nhead = 4 if trans_dim >= 64 else 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=nhead,
            dim_feedforward=trans_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.post_trans = nn.Sequential(
            nn.Linear(trans_dim, trans_dim),
            nn.ReLU()
        )

        # ABMIL
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)

        self.risk_predictor = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim)
        )

        initialize_weights(self)

    def forward(self, x, patch_batch_size=256, device=None):
        """
        x: [B, S, N, D]
        """
        if device is not None:
            x = x.to(device)

        B, S, N, D = x.shape
        assert S == self.num_models

        # [B, N, S, D]
        x = x.permute(0, 2, 1, 3)

        fused = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)

        for b in range(B):
            for start in range(0, N, patch_batch_size):
                chunk = x[b, start:start+patch_batch_size]  # [n, S, D]
                trans_out = self.transformer(chunk)
                fused[b, start:start+patch_batch_size] = self.post_trans(
                    trans_out.mean(dim=1)
                )

        if self.selector is not None:
            fused, _ = self.selector(fused)

        if self.is_batch:
            return self._forward_patch_batch(fused, patch_batch_size)
        else:
            return self._forward_single_batch(fused)

    # ===== ABMIL 与 Global 版完全一致 =====
    def _forward_single_batch(self, H):
        B, N, D = H.shape
        risk_scores, attn_list, Z_list = [], [], []

        for i in range(B):
            Hi = H[i]
            V = self.attention_V(Hi)
            U = self.attention_U(Hi)
            A = F.softmax(self.attention_w(V * U).t(), dim=1)
            Z = A @ Hi
            risk = self.risk_predictor(Z).view(-1)
            attn_list.append(A)
            Z_list.append(Z)
            risk_scores.append(risk)

        return (attn_list, Z_list), torch.stack(risk_scores)

    def _forward_patch_batch(self, x, patch_batch_size=256):
        B, N, D = x.shape
        attn_list, Z_list, risk_list = [], [], []

        for b in range(B):
            xb = x[b]
            logits_all, weighted_sum, sum_expA = [], None, None

            for start in range(0, N, patch_batch_size):
                xp = xb[start:start+patch_batch_size]
                V = self.attention_V(xp)
                U = self.attention_U(xp)
                logits = self.attention_w(V * U)
                logits_all.append(logits.detach().cpu())

                maxv = logits.max()
                expA = torch.exp(logits - maxv)
                sum_exp = expA.sum(dim=0)
                weighted = (expA.unsqueeze(-1) * xp.unsqueeze(1)).sum(dim=0)

                sum_expA = sum_exp if sum_expA is None else sum_expA + sum_exp
                weighted_sum = weighted if weighted_sum is None else weighted_sum + weighted

            Z = weighted_sum / (sum_expA.unsqueeze(-1) + 1e-8)
            risk = self.risk_predictor(Z.view(1, -1)).view(-1)

            A = F.softmax(torch.cat(logits_all, dim=0).t().to(x.device), dim=1)

            attn_list.append(A)
            Z_list.append(Z)
            risk_list.append(risk)

        return (attn_list, Z_list), torch.stack(risk_list)
