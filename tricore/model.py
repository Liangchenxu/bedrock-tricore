from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def tricore_fused_scan(
    u_feat: torch.Tensor,
    u_surp_bias: torch.Tensor,
    w_gate_logits: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    base_decay: torch.Tensor,
    fast_decay: torch.Tensor,
    fast_lr: torch.Tensor,
    hop_matrices: torch.Tensor,
    hop_weights: torch.Tensor,
    enable_surprise_gate: bool,
    enable_fast_weight: bool,
    enable_adaptive_loop: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    High-Speed JIT Fused Scan with zero ``F_seq`` materialization.

    Engine I and Engine II evolve the O(1) recurrent state in causal order.
    Engine III's fixed multi-hop re-query is fused into the same loop, using the
    current ``F_mat`` for each timestep.  Only the fused memory read
    ``[B, L, H * Dh]`` is stored; no ``[B, L, H, Dh, Dh]`` snapshot tensor is
    allocated or stacked.
    """
    B, L, D = u_feat.shape
    H = Q.shape[2]
    Dh = Q.shape[3]

    h = torch.zeros(B, D, device=u_feat.device, dtype=u_feat.dtype)
    F_mat = torch.zeros(B, H, Dh, Dh, device=u_feat.device, dtype=u_feat.dtype)
    prev_k = torch.zeros(B, H, Dh, 1, device=u_feat.device, dtype=u_feat.dtype)

    h_list = torch.jit.annotate(list[torch.Tensor], [])
    r1_list = torch.jit.annotate(list[torch.Tensor], [])
    mem_list = torch.jit.annotate(list[torch.Tensor], [])
    surprise_list = torch.jit.annotate(list[torch.Tensor], [])

    for t in range(L):
        u_t = u_feat[:, t, :]

        # 1. Engine I: Surprise-Gated State Collapse & Write Protection
        pred_error = torch.abs(u_t - h)
        if enable_surprise_gate:
            g_t = torch.sigmoid(u_surp_bias[:, t, :] + pred_error)
            surprise_scalar = g_t.mean(dim=-1, keepdim=True)
            write_gate = torch.sigmoid(w_gate_logits[:, t] + surprise_scalar.view(B, 1, 1, 1) - 0.5)
            eff_lr = fast_lr * write_gate
            eff_decay = fast_decay
        else:
            # Ungated ablation: constant state decay & unshielded overwrite with forgetting
            g_t = torch.full_like(u_t, 0.5)
            surprise_scalar = g_t.mean(dim=-1, keepdim=True)
            eff_lr = torch.ones_like(fast_lr) * 0.95
            eff_decay = torch.ones_like(fast_decay) * 0.82

        h = base_decay * (1.0 - 0.5 * g_t) * h + g_t * u_t
        h_list.append(h)
        surprise_list.append(surprise_scalar)

        # 2. Engine II: Causal Shifted Fast-Weight Update & Hop-1 Read
        if enable_fast_weight:
            k_t = K[:, t].unsqueeze(-1)
            v_t = V[:, t].unsqueeze(-1)
            q_t = Q[:, t].unsqueeze(-1)

            if t > 0:
                v_pred = torch.matmul(F_mat, prev_k)
                synapse_err = v_t - v_pred
                F_mat = eff_decay * F_mat + torch.matmul(eff_lr * synapse_err, prev_k.transpose(-1, -2))

            r_hop1 = torch.matmul(F_mat, q_t)
            if enable_adaptive_loop:
                # Engine III: fixed multi-hop refinement using this timestep's
                # F_mat.  This is identical to forward_step's recurrence.
                r_accum = hop_weights[:, t, 0].view(B, 1, 1, 1) * r_hop1
                r_curr = r_hop1
                for step in range(hop_matrices.shape[0]):
                    q_next = torch.nn.functional.normalize(
                        r_curr + torch.matmul(hop_matrices[step], r_curr),
                        p=2.0,
                        dim=-2,
                    )
                    r_curr = torch.matmul(F_mat, q_next)
                    r_accum = r_accum + hop_weights[
                        :, t, step + 1
                    ].view(B, 1, 1, 1) * r_curr
                mem_list.append(r_accum)
            else:
                r1_list.append(r_hop1)
            prev_k = k_t

    h_seq = torch.stack(h_list, dim=1)
    surprise_trace = torch.cat(surprise_list, dim=1)

    if enable_fast_weight:
        if enable_adaptive_loop:
            mem_seq = torch.stack(mem_list, dim=1).squeeze(-1).reshape(
                B, L, H * Dh
            )
        else:
            mem_seq = torch.stack(r1_list, dim=1).squeeze(-1).reshape(
                B, L, H * Dh
            )
    else:
        mem_seq = torch.zeros(
            B, L, H * Dh, device=u_feat.device, dtype=u_feat.dtype
        )

    return h_seq, mem_seq, surprise_trace


class TriCoreBlock(nn.Module):
    """
    TriCore V9: Hybrid JIT-Recurrent + Multi-Hop Equilibrium Operator
    with native O(1) per-token streaming inference.

    Engine III's re-query is fused into the JIT scan, so training never
    materializes the ``F_seq: [B, L, H, d_head, d_head]`` snapshot tensor.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_head: int = 32, max_loops: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_loops = max_loops
        total_fast_dim = n_heads * d_head

        self.in_norm = nn.LayerNorm(d_model)

        # --- Engine I: Surprise-Gated State Collapse & Write Protection ---
        self.u_proj = nn.Linear(d_model, d_model)
        self.surprise_u_proj = nn.Linear(d_model, d_model)
        self.write_gate_proj = nn.Linear(d_model, n_heads)
        self.base_decay_logit = nn.Parameter(torch.linspace(1.0, 3.0, d_model))

        # --- Engine II: Subspace-Aligned Multi-Head Fast-Weight Plasticity ---
        self.proj_addr = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_q_res = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_k_res = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_v_res = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_out = nn.Linear(total_fast_dim, d_model, bias=False)

        self.fast_lr_logit = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 2.2)
        self.fast_decay_logit = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 4.5)

        # --- Engine III: Vectorized Multi-Hop Synaptic Equilibrium Loop ---
        self.hop_proj = nn.Parameter(torch.randn(max_loops, 1, 1, n_heads, d_head, d_head) * 0.02)
        self.hop_router = nn.Linear(d_model, max_loops + 1)

        self.think_norm = nn.LayerNorm(d_model)
        self.think_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """Create the O(1) streaming state for one independent sequence."""
        h = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        F_mat = torch.zeros(
            batch_size, self.n_heads, self.d_head, self.d_head,
            device=device, dtype=dtype,
        )
        prev_k = torch.zeros(
            batch_size, self.n_heads, self.d_head, 1,
            device=device, dtype=dtype,
        )
        return h, F_mat, prev_k

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: tuple,
        step_idx: int,
        enable_surprise_gate: bool = True,
        enable_fast_weight: bool = True,
        enable_adaptive_loop: bool = True,
    ):
        """Run one causal token through the block while carrying O(1) state.

        This is an exact single-token version of ``forward``.  It never
        materializes the training-only ``F_seq`` tensor: the only persistent
        recurrent tensors are ``h``, ``F_mat`` and ``prev_k``.
        """
        if x_t.dim() == 3:
            if x_t.shape[1] != 1:
                raise ValueError("forward_step expects x_t with shape [B, D] or [B, 1, D]")
            x_t = x_t[:, 0, :]
        elif x_t.dim() != 2:
            raise ValueError("forward_step expects x_t with shape [B, D] or [B, 1, D]")

        h, F_mat, prev_k = state
        B, D = x_t.shape
        if h.shape != (B, D):
            raise ValueError(f"state h shape {tuple(h.shape)} does not match input {(B, D)}")

        H, Dh = self.n_heads, self.d_head
        residual = x_t
        u = self.in_norm(x_t)

        u_feat = F.silu(self.u_proj(u))
        u_surp_bias = self.surprise_u_proj(u)
        w_gate_logits = self.write_gate_proj(u).view(B, H, 1, 1)

        addr = F.normalize(self.proj_addr(u).view(B, H, Dh), p=2.0, dim=-1)
        Q = F.normalize(
            addr + 0.15 * self.proj_q_res(u).view(B, H, Dh), p=2.0, dim=-1
        )
        if enable_surprise_gate:
            K = F.normalize(
                addr + 0.15 * self.proj_k_res(u).view(B, H, Dh), p=2.0, dim=-1
            )
            V = addr + 0.15 * self.proj_v_res(u).view(B, H, Dh)
        else:
            K = addr
            V = addr

        base_decay = torch.sigmoid(self.base_decay_logit)
        fast_decay = torch.sigmoid(self.fast_decay_logit)
        fast_lr = torch.sigmoid(self.fast_lr_logit)

        # Engine I: one causally shifted state update.
        pred_error = torch.abs(u_feat - h)
        if enable_surprise_gate:
            g_t = torch.sigmoid(u_surp_bias + pred_error)
            surprise_t = g_t.mean(dim=-1, keepdim=True)
            write_gate = torch.sigmoid(
                w_gate_logits + surprise_t.view(B, 1, 1, 1) - 0.5
            )
            eff_lr = fast_lr * write_gate
            eff_decay = fast_decay
        else:
            g_t = torch.full_like(u_feat, 0.5)
            surprise_t = g_t.mean(dim=-1, keepdim=True)
            eff_lr = torch.ones_like(fast_lr) * 0.95
            eff_decay = torch.ones_like(fast_decay) * 0.82

        h_new = base_decay * (1.0 - 0.5 * g_t) * h + g_t * u_feat

        # Engine II: causal-shift fast-weight update and hop-1 read.
        F_new = F_mat
        prev_k_new = prev_k
        r_hop1 = torch.zeros(B, H, Dh, 1, device=x_t.device, dtype=x_t.dtype)
        if enable_fast_weight:
            k_t = K.unsqueeze(-1)
            v_t = V.unsqueeze(-1)
            q_t = Q.unsqueeze(-1)
            if step_idx > 0:
                v_pred = torch.matmul(F_mat, prev_k)
                synapse_err = v_t - v_pred
                F_new = eff_decay * F_mat + torch.matmul(
                    eff_lr * synapse_err, prev_k.transpose(-1, -2)
                )
            r_hop1 = torch.matmul(F_new, q_t)
            prev_k_new = k_t

        # Engine III: the same fixed multi-hop refinement as ``forward``.
        if enable_fast_weight and enable_adaptive_loop:
            hop_weights = F.softmax(self.hop_router(u), dim=-1)
            r_accum = hop_weights[:, 0].view(B, 1, 1, 1) * r_hop1
            r_curr = r_hop1
            hop_matrices = self.hop_proj.view(
                self.max_loops, self.n_heads, self.d_head, self.d_head
            )
            for hop in range(self.max_loops):
                q_next = F.normalize(
                    r_curr + torch.matmul(hop_matrices[hop], r_curr),
                    p=2.0,
                    dim=-2,
                )
                r_curr = torch.matmul(F_new, q_next)
                r_accum = r_accum + hop_weights[:, hop + 1].view(B, 1, 1, 1) * r_curr
            mem_t = r_accum.squeeze(-1).reshape(B, H * Dh)
            z = h_new + self.proj_out(mem_t)
        elif enable_fast_weight:
            mem_t = r_hop1.squeeze(-1).reshape(B, H * Dh)
            z = h_new + self.proj_out(mem_t)
        else:
            z = h_new

        if enable_adaptive_loop:
            z = z + surprise_t * self.think_mlp(self.think_norm(z))

        out_t = residual + self.out_norm(z)
        return out_t, surprise_t, (h_new, F_new, prev_k_new)

    def forward(
        self,
        x: torch.Tensor,
        enable_surprise_gate: bool = True,
        enable_fast_weight: bool = True,
        enable_adaptive_loop: bool = True,
    ):
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        residual = x
        u = self.in_norm(x)

        u_feat = F.silu(self.u_proj(u))
        u_surp_bias = self.surprise_u_proj(u)
        w_gate_logits = self.write_gate_proj(u).view(B, L, H, 1, 1)
        hop_weights = F.softmax(self.hop_router(u), dim=-1)  # [B, L, max_loops + 1]

        addr = F.normalize(self.proj_addr(u).view(B, L, H, Dh), p=2.0, dim=-1)
        Q = F.normalize(addr + 0.15 * self.proj_q_res(u).view(B, L, H, Dh), p=2.0, dim=-1)
        if enable_surprise_gate:
            K = F.normalize(addr + 0.15 * self.proj_k_res(u).view(B, L, H, Dh), p=2.0, dim=-1)
            V = addr + 0.15 * self.proj_v_res(u).view(B, L, H, Dh)
        else:
            # Without Engine I surprise-aware residual modulation, keys & values use raw canonical subspace
            K = addr
            V = addr

        base_decay = torch.sigmoid(self.base_decay_logit)
        fast_decay = torch.sigmoid(self.fast_decay_logit)
        fast_lr = torch.sigmoid(self.fast_lr_logit)
        hop_matrices = self.hop_proj.view(
            self.max_loops, H, Dh, Dh
        )

        # Steps 1-3: one fused JIT scan.  Engine III runs inside the causal
        # loop with the current F_mat and returns only the memory read, never
        # an F_seq snapshot tensor.
        h_seq, mem_seq, surprise_trace = tricore_fused_scan(
            u_feat=u_feat,
            u_surp_bias=u_surp_bias,
            w_gate_logits=w_gate_logits,
            Q=Q,
            K=K,
            V=V,
            base_decay=base_decay,
            fast_decay=fast_decay,
            fast_lr=fast_lr,
            hop_matrices=hop_matrices,
            hop_weights=hop_weights,
            enable_surprise_gate=enable_surprise_gate,
            enable_fast_weight=enable_fast_weight,
            enable_adaptive_loop=enable_adaptive_loop,
        )

        if enable_fast_weight:
            z = h_seq + self.proj_out(mem_seq)
        else:
            z = h_seq

        if enable_adaptive_loop:
            s_expanded = surprise_trace.unsqueeze(-1)
            z = z + s_expanded * self.think_mlp(self.think_norm(z))

        out = residual + self.out_norm(z)
        return out, surprise_trace


class TriCoreLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 1,
        d_fast: int = 32,
        max_loops: int = 2,
        enable_surprise_gate: bool = True,
        enable_fast_weight: bool = True,
        enable_adaptive_loop: bool = True,
    ):
        super().__init__()
        self.enable_surprise_gate = enable_surprise_gate
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
                enable_surprise_gate=self.enable_surprise_gate,
                enable_fast_weight=self.enable_fast_weight,
                enable_adaptive_loop=self.enable_adaptive_loop,
            )
            surprises.append(s)
        logits = self.lm_head(self.final_norm(x))
        mean_surprise = torch.stack(surprises, dim=0).mean(dim=0)
        return logits, mean_surprise

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """Create one O(1) state tuple per TriCore layer."""
        return tuple(
            layer.init_state(batch_size, device, dtype) for layer in self.layers
        )

    def forward_step(self, idx_t: torch.Tensor, states, step_idx: int):
        """Decode one token and return ``(logits, surprise, new_states)``.

        ``states`` is the tuple returned by :meth:`init_state`.  For
        convenience, a single block state ``(h, F_mat, prev_k)`` is also
        accepted when the model has exactly one layer.
        """
        if idx_t.dim() == 2:
            if idx_t.shape[1] != 1:
                raise ValueError("forward_step expects idx_t with shape [B] or [B, 1]")
            idx_t = idx_t[:, 0]
        elif idx_t.dim() != 1:
            raise ValueError("forward_step expects idx_t with shape [B] or [B, 1]")

        single_block_state = (
            len(self.layers) == 1
            and isinstance(states, (tuple, list))
            and len(states) == 3
            and torch.is_tensor(states[0])
        )
        layer_states = (tuple(states),) if single_block_state else tuple(states)
        if len(layer_states) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} layer states, got {len(layer_states)}"
            )

        x = self.embedding(idx_t)
        surprises = []
        new_layer_states = []
        for layer, state in zip(self.layers, layer_states):
            x, surprise_t, new_state = layer.forward_step(
                x,
                state,
                step_idx,
                enable_surprise_gate=self.enable_surprise_gate,
                enable_fast_weight=self.enable_fast_weight,
                enable_adaptive_loop=self.enable_adaptive_loop,
            )
            surprises.append(surprise_t)
            new_layer_states.append(new_state)

        logits = self.lm_head(self.final_norm(x))
        mean_surprise = torch.stack(surprises, dim=0).mean(dim=0)
        new_states = tuple(new_layer_states)
        if single_block_state:
            new_states = new_states[0]
        return logits, mean_surprise, new_states


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


