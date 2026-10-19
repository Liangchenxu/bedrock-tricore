import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriCoreBlock(nn.Module):
    """
    TriCore V3: High-Speed Unified Operator with Causal Shifted Fast-Weight Binding
      1. Engine I: Surprise-Gated State Collapse (Vectorized pre-projection + O(1) recurrent state)
      2. Engine II: Multi-Head Forward Fast-Weight Plasticity (Causal K_{t-1} -> V_t Delta Rule + Current Step Recall)
      3. Engine III: Surprise-Driven Adaptive Equilibrium Loop (Vectorized post-memory refinement)
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_head: int = 32, max_loops: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_loops = max_loops

        self.in_norm = nn.LayerNorm(d_model)

        # Causal 1D Short Convolution for local temporal context
        self.short_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=2,
            groups=d_model,
        )

        # --- Engine I: Surprise-Gated State Collapse ---
        self.u_proj = nn.Linear(d_model, d_model)
        self.surprise_gate = nn.Linear(d_model, d_model)
        self.base_decay_logit = nn.Parameter(torch.linspace(1.0, 3.0, d_model))

        # --- Engine II: Multi-Head Causal Fast-Weight Plasticity ---
        total_fast_dim = n_heads * d_head
        self.proj_q = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_k = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_v = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_out = nn.Linear(total_fast_dim, d_model, bias=False)

        self.fast_lr_logit = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 0.5)
        self.fast_decay_logit = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 3.5)

        # --- Engine III: Adaptive Nonlinear Equilibrium Loop ---
        self.think_norm = nn.LayerNorm(d_model)
        self.think_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enable_fast_weight: bool = True,
        enable_adaptive_loop: bool = True,
    ):
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head
        device = x.device

        residual = x
        u = self.in_norm(x)

        # Pre-compute local conv features in parallel
        u_conv = self.short_conv(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u_conv = F.silu(u_conv)
        u_feat = self.u_proj(u_conv)

        # Multi-head Q, K, V projections computed in parallel across sequence L
        # Q and K use sharp L2 normalization along head dim so dot-product retrieval is exact
        Q = F.normalize(self.proj_q(u).view(B, L, H, Dh), p=2, dim=-1)
        K = F.normalize(self.proj_k(u).view(B, L, H, Dh), p=2, dim=-1)
        V = self.proj_v(u_conv).view(B, L, H, Dh)

        base_decay = torch.sigmoid(self.base_decay_logit)               # [D]
        fast_decay = torch.sigmoid(self.fast_decay_logit)               # [1, H, 1, 1]
        fast_lr = torch.sigmoid(self.fast_lr_logit) * 0.9               # [1, H, 1, 1]

        # Persistent O(1) states
        h = torch.zeros(B, D, device=device, dtype=x.dtype)
        # Multi-head fast weight matrix F_mat: [B, H, Dh, Dh]
        F_mat = torch.zeros(B, H, Dh, Dh, device=device, dtype=x.dtype)
        prev_k = torch.zeros(B, H, Dh, 1, device=device, dtype=x.dtype)

        h_list = []
        mem_list = []
        surprise_list = []

        for t in range(L):
            u_t = u_feat[:, t, :]  # [B, D]

            # 1. Engine I: Surprise-Gated State Collapse (Prediction Error against compressed state h)
            pred_error = torch.abs(u_t - h)
            g_t = torch.sigmoid(self.surprise_gate(pred_error))  # [B, D]
            h = base_decay * (1.0 - 0.5 * g_t) * h + g_t * u_t
            h_list.append(h.unsqueeze(1))

            surprise_scalar = g_t.mean(dim=-1, keepdim=True)  # [B, 1]
            surprise_list.append(surprise_scalar)

            # 2. Engine II: Causal Shifted Fast-Weight Plasticity (Bind K_{t-1} -> V_t, then Recall with Q_t)
            if enable_fast_weight:
                k_t = K[:, t].unsqueeze(-1)  # [B, H, Dh, 1]
                v_t = V[:, t].unsqueeze(-1)  # [B, H, Dh, 1]
                q_t = Q[:, t].unsqueeze(-1)  # [B, H, Dh, 1]

                # Causal Associative Write: bind previous token's Key (prev_k) to current token's Value (v_t)
                if t > 0:
                    v_pred = torch.matmul(F_mat, prev_k)  # [B, H, Dh, 1]
                    synapse_err = v_t - v_pred
                    s_gate = surprise_scalar.view(B, 1, 1, 1)
                    eff_lr = fast_lr * (0.3 + 0.7 * s_gate)
                    # Delta rule outer-product update: [B, H, Dh, 1] x [B, H, 1, Dh] -> [B, H, Dh, Dh]
                    F_mat = fast_decay * F_mat + torch.matmul(eff_lr * synapse_err, prev_k.transpose(-1, -2))

                # Associative Read: query the updated synaptic matrix using current token's Query (q_t)
                recalled = torch.matmul(F_mat, q_t).squeeze(-1).reshape(B, H * Dh)
                mem_list.append(recalled.unsqueeze(1))
                prev_k = k_t

        h_seq = torch.cat(h_list, dim=1)                  # [B, L, D]
        surprise_trace = torch.cat(surprise_list, dim=1)  # [B, L]

        if enable_fast_weight:
            mem_seq = torch.cat(mem_list, dim=1)          # [B, L, H*Dh]
            z = h_seq + self.proj_out(mem_seq)
        else:
            z = h_seq

        # 3. Engine III: Surprise-Driven Adaptive Equilibrium Loop (Vectorized across L for GPU speed!)
        if enable_adaptive_loop:
            s_expanded = surprise_trace.unsqueeze(-1)     # [B, L, 1]
            for step in range(self.max_loops):
                step_scale = s_expanded / float(step + 1)
                z = z + step_scale * self.think_mlp(self.think_norm(z))

        out = residual + self.out_norm(z)
        return out, surprise_trace


class TriCoreLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        d_fast: int = 32,
        max_loops: int = 2,
        enable_fast_weight: bool = True,
        enable_adaptive_loop: bool = True,
    ):
        super().__init__()
        self.enable_fast_weight = enable_fast_weight
        self.enable_adaptive_loop = enable_adaptive_loop
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TriCoreBlock(d_model=d_model, n_heads=4, d_head=d_fast, max_loops=max_loops)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        x = self.embedding(idx)
        surprises = []
        for layer in self.layers:
            x, s = layer(
                x,
                enable_fast_weight=self.enable_fast_weight,
                enable_adaptive_loop=self.enable_adaptive_loop,
            )
            surprises.append(s)
        logits = self.lm_head(self.final_norm(x))
        mean_surprise = torch.stack(surprises, dim=0).mean(dim=0)
        return logits, mean_surprise


class BaselineTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, max_seq_len: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        B, L = idx.shape
        positions = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.embedding(idx) + self.pos_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=idx.device)
        x = self.encoder(x, mask=causal_mask, is_causal=True)
        logits = self.lm_head(self.final_norm(x))
        return logits, None
