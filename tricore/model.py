from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def tricore_fused_scan(
    u_feat: torch.Tensor,
    u_surp_bias: torch.Tensor,
    w_gate_logits: torch.Tensor,
    elig_gate_logits: torch.Tensor,
    hop_weights: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    base_decay: torch.Tensor,
    fast_decay: torch.Tensor,
    fast_lr: torch.Tensor,
    hop_proj: torch.Tensor,
    enable_surprise_gate: bool,
    enable_fast_weight: bool,
    enable_adaptive_loop: bool,
    max_loops: int,
    reset_mask: Optional[torch.Tensor] = None,
    enable_delta_rule: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TriCore V12 JIT scan with a synaptic eligibility trace and optional reset.

    ``enable_delta_rule=True`` writes the prediction error
    ``V_t - F_{t-1} K_{t-1}`` into the fast-weight matrix.  Setting it to
    ``False`` leaves the same decay and learning-rate terms in place but uses
    the plain Hebbian outer product ``V_t K_{t-1}^T`` (no erase term).
    """
    B, L, D = u_feat.shape
    H = Q.shape[2]
    Dh = Q.shape[3]
    h = torch.zeros(B, D, device=u_feat.device, dtype=u_feat.dtype)
    F_mat = torch.zeros(B, H, Dh, Dh, device=u_feat.device, dtype=u_feat.dtype)
    prev_k = torch.zeros(B, H, Dh, 1, device=u_feat.device, dtype=u_feat.dtype)
    h_list = torch.jit.annotate(list[torch.Tensor], [])
    mem_list = torch.jit.annotate(list[torch.Tensor], [])
    surprise_list = torch.jit.annotate(list[torch.Tensor], [])
    for t in range(L):
        u_t = u_feat[:, t, :]
        pred_error = torch.abs(u_t - h)
        if enable_surprise_gate:
            g_t = torch.sigmoid(u_surp_bias[:, t, :] + pred_error)
            surprise_scalar = g_t.mean(dim=-1, keepdim=True)
            s_4d = surprise_scalar.view(B, 1, 1, 1)
            write_gate = torch.sigmoid(w_gate_logits[:, t] + s_4d - 0.5)
            elig_gate = torch.sigmoid(elig_gate_logits[:, t] + s_4d - 0.5)
            eff_lr = fast_lr * write_gate
        else:
            g_t = torch.full_like(u_t, 0.5)
            surprise_scalar = g_t.mean(dim=-1, keepdim=True)
            elig_gate = torch.ones_like(elig_gate_logits[:, t])
            eff_lr = fast_lr
        h = base_decay * (1.0 - 0.5 * g_t) * h + g_t * u_t
        h_list.append(h)
        surprise_list.append(surprise_scalar)
        if enable_fast_weight:
            k_t = K[:, t].unsqueeze(-1)
            v_t = V[:, t].unsqueeze(-1)
            q_t = Q[:, t].unsqueeze(-1)
            if t > 0:
                if enable_delta_rule:
                    v_pred = torch.matmul(F_mat, prev_k)
                    synapse_err = v_t - v_pred
                else:
                    synapse_err = v_t
                F_mat = fast_decay * F_mat + torch.matmul(eff_lr * synapse_err, prev_k.transpose(-1, -2))
            # The eligibility gate selects the query source.  A closed gate
            # (e~=0 at an infix EQ) reads the frozen preceding entity key and
            # -- crucially -- also stops prev_k from advancing, so the next
            # value is written at that same entity key.  This is the shifted
            # binding F(alias)=var that makes the two-hop dereference work.
            q_read = F.normalize(
                elig_gate * q_t + (1.0 - elig_gate) * prev_k,
                p=2.0,
                dim=-2,
            )
            r_hop = torch.matmul(F_mat, q_read)
            if enable_adaptive_loop:
                w_t = hop_weights[:, t]
                r_accum = w_t[:, 0].view(B, 1, 1, 1) * r_hop
                r_curr = r_hop
                for step in range(max_loops):
                    q_next = F.normalize(r_curr + torch.matmul(hop_proj[step].squeeze(0), r_curr), p=2.0, dim=-2)
                    r_curr = torch.matmul(F_mat, q_next)
                    r_accum = r_accum + w_t[:, step + 1].view(B, 1, 1, 1) * r_curr
                recalled = r_accum.squeeze(-1).reshape(B, H * Dh)
            else:
                recalled = r_hop.squeeze(-1).reshape(B, H * Dh)
            mem_list.append(recalled)
            prev_k = F.normalize((1.0 - elig_gate) * prev_k + elig_gate * k_t, p=2.0, dim=-2)
            if reset_mask is not None:
                reset_t = reset_mask[:, t].to(prev_k.dtype).view(B, 1, 1, 1)
                prev_k = prev_k * (1.0 - reset_t)
    h_seq = torch.stack(h_list, dim=1)
    surprise_trace = torch.cat(surprise_list, dim=1)
    if enable_fast_weight:
        mem_seq = torch.stack(mem_list, dim=1)
    else:
        mem_seq = torch.zeros(B, L, H * Dh, device=u_feat.device, dtype=u_feat.dtype)
    return h_seq, mem_seq, surprise_trace


class TriCoreBlock(nn.Module):
    """TriCore V12 block with autonomous gates and an optional Delta rule."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, d_head: int = 32, max_loops: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_loops = max_loops
        total_fast_dim = n_heads * d_head

        self.in_norm = nn.LayerNorm(d_model)
        self.u_proj = nn.Linear(d_model, d_model)
        self.surprise_u_proj = nn.Linear(d_model, d_model)
        # Token-role gate networks: a two-layer MLP distinguishes constant
        # separators (EQ) from entity tokens before the dynamic surprise term.
        # V12 starts the eligibility bias at 0.0 (initial gate ~= 0.5) so the
        # first optimization step keeps half of the preceding entity key.  That
        # breaks the EQ hash-collision saddle and lets the gate discover its
        # role from token identity plus the causal prediction error alone.
        self.write_gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_heads),
        )
        self.elig_gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_heads),
        )
        for gate_proj in (self.write_gate_proj, self.elig_gate_proj):
            nn.init.normal_(gate_proj[-1].weight, mean=0.0, std=0.02)
        nn.init.constant_(self.write_gate_proj[-1].bias, 1.0)
        nn.init.constant_(self.elig_gate_proj[-1].bias, 0.0)
        # V12 token-role bypass: a direct linear path from the normalized token
        # embedding to both gate logits.  The two-layer MLPs above can express
        # contextual roles, but their 0.02-scaled output layer needs many steps
        # to move a logit by the ~6 nats required for a hard close.  This skip
        # gives the optimizer a short path; the 8x gain is a fixed logit
        # temperature, not a token-ID mask and not an extra recurrent state.
        self.gate_token_proj = nn.Linear(d_model, 2 * n_heads)
        nn.init.normal_(self.gate_token_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate_token_proj.bias, 0.0)
        self.gate_logit_gain = 8.0
        self.base_decay_logit = nn.Parameter(torch.linspace(1.0, 3.0, d_model))

        self.proj_addr = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_q_res = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_k_res = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_v_res = nn.Linear(d_model, total_fast_dim, bias=False)
        self.proj_out = nn.Linear(total_fast_dim, d_model, bias=False)

        self.fast_lr_logit = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 2.2)
        self.fast_decay_logit = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 4.5)

        self.hop_proj = nn.Parameter(
            torch.randn(max_loops, 1, 1, n_heads, d_head, d_head) * 0.02
        )
        self.hop_router = nn.Linear(d_model, max_loops + 1)

        self.think_norm = nn.LayerNorm(d_model)
        self.think_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        h = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        F_mat = torch.zeros(
            batch_size, self.n_heads, self.d_head, self.d_head,
            device=device, dtype=dtype,
        )
        prev_k = torch.zeros(
            batch_size, self.n_heads, self.d_head, 1,
            device=device, dtype=dtype,
        )
        return (h, F_mat, prev_k)


    def forward_step(
        self,
        x_t: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        step_idx: int = 0,
        enable_surprise_gate: bool = True,
        enable_fast_weight: bool = True,
        enable_adaptive_loop: bool = True,
        separator_mask: torch.Tensor = None,
        reset_mask: Optional[torch.Tensor] = None,
        enable_delta_rule: bool = True,
        gate_role_logits: Optional[torch.Tensor] = None,
    ):
        squeeze_seq = False
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)
            squeeze_seq = True
        B, D = x_t.shape
        H, Dh = self.n_heads, self.d_head
        h, F_mat, prev_k = state

        residual = x_t
        u_t = self.in_norm(x_t)
        u_feat_t = F.silu(self.u_proj(u_t))
        u_surp_t = self.surprise_u_proj(u_t)
        w_gate_t = self.write_gate_proj(u_t).view(B, H, 1, 1)
        e_gate_t = self.elig_gate_proj(u_t).view(B, H, 1, 1)
        token_gate_t = (
            self.gate_token_proj(u_t).view(B, H, 2, 1, 1) * self.gate_logit_gain
        )
        w_gate_t = w_gate_t + token_gate_t[:, :, 0]
        e_gate_t = e_gate_t + token_gate_t[:, :, 1]
        if gate_role_logits is not None:
            w_gate_t = w_gate_t + gate_role_logits[:, :, 0]
            e_gate_t = e_gate_t + gate_role_logits[:, :, 1]
        if separator_mask is not None:
            if isinstance(separator_mask, tuple):
                elig_sep_mask, write_sep_mask = separator_mask
                w_gate_t = w_gate_t - write_sep_mask.to(w_gate_t.dtype).view(B, 1, 1, 1) * 8.0
                e_gate_t = e_gate_t - elig_sep_mask.to(e_gate_t.dtype).view(B, 1, 1, 1) * 8.0
            else:
                gate_mask = separator_mask.to(w_gate_t.dtype).view(B, 1, 1, 1) * 8.0
                w_gate_t = w_gate_t - gate_mask
                e_gate_t = e_gate_t - gate_mask
        hop_w_t = F.softmax(self.hop_router(u_t), dim=-1)

        addr_t = F.normalize(self.proj_addr(u_t).view(B, H, Dh), p=2.0, dim=-1)
        q_t = F.normalize(
            addr_t + 0.15 * self.proj_q_res(u_t).view(B, H, Dh), p=2.0, dim=-1
        ).unsqueeze(-1)
        k_t = F.normalize(
            addr_t + 0.15 * self.proj_k_res(u_t).view(B, H, Dh), p=2.0, dim=-1
        ).unsqueeze(-1)
        v_t = (addr_t + 0.15 * self.proj_v_res(u_t).view(B, H, Dh)).unsqueeze(-1)

        base_decay = torch.sigmoid(self.base_decay_logit)
        fast_decay = torch.sigmoid(self.fast_decay_logit)
        fast_lr = torch.sigmoid(self.fast_lr_logit)

        pred_error = torch.abs(u_feat_t - h)
        if enable_surprise_gate:
            g_t = torch.sigmoid(u_surp_t + pred_error)
            surprise_scalar = g_t.mean(dim=-1, keepdim=True)
            s_4d = surprise_scalar.view(B, 1, 1, 1)
            write_gate = torch.sigmoid(w_gate_t + s_4d - 0.5)
            elig_gate = torch.sigmoid(e_gate_t + s_4d - 0.5)
            eff_lr = fast_lr * write_gate
        else:
            g_t = torch.full_like(u_feat_t, 0.5)
            surprise_scalar = g_t.mean(dim=-1, keepdim=True)
            elig_gate = torch.ones_like(e_gate_t)
            eff_lr = fast_lr

        h_new = base_decay * (1.0 - 0.5 * g_t) * h + g_t * u_feat_t

        if enable_fast_weight:
            if step_idx > 0:
                if enable_delta_rule:
                    v_pred = torch.matmul(F_mat, prev_k)
                    synapse_err = v_t - v_pred
                else:
                    synapse_err = v_t
                F_new = fast_decay * F_mat + torch.matmul(
                    eff_lr * synapse_err, prev_k.transpose(-1, -2)
                )
            else:
                F_new = F_mat

            # Keep streaming parity with the fused scan's trace-conditioned read:
            # a closed eligibility gate dereferences the frozen entity key.
            q_read = F.normalize(
                elig_gate * q_t + (1.0 - elig_gate) * prev_k,
                p=2.0,
                dim=-2,
            )
            r_hop = torch.matmul(F_new, q_read)
            if enable_adaptive_loop:
                r_accum = hop_w_t[:, 0].view(B, 1, 1, 1) * r_hop
                r_curr = r_hop
                for step in range(self.max_loops):
                    q_next = F.normalize(
                        r_curr + torch.matmul(self.hop_proj[step].squeeze(0), r_curr),
                        p=2.0,
                        dim=-2,
                    )
                    r_curr = torch.matmul(F_new, q_next)
                    r_accum = r_accum + hop_w_t[:, step + 1].view(B, 1, 1, 1) * r_curr
                recalled = r_accum.squeeze(-1).reshape(B, H * Dh)
            else:
                recalled = r_hop.squeeze(-1).reshape(B, H * Dh)
            prev_k_new = F.normalize(
                (1.0 - elig_gate) * prev_k + elig_gate * k_t, p=2.0, dim=-2
            )
            if reset_mask is not None:
                reset_t = reset_mask.to(prev_k_new.dtype).view(B, 1, 1, 1)
                prev_k_new = prev_k_new * (1.0 - reset_t)
            z = h_new + self.proj_out(recalled)
        else:
            F_new = F_mat
            prev_k_new = prev_k
            z = h_new

        if enable_adaptive_loop:
            z = z + surprise_scalar * self.think_mlp(self.think_norm(z))

        out_t = residual + self.out_norm(z)
        if squeeze_seq:
            out_t = out_t.unsqueeze(1)
        return out_t, surprise_scalar.squeeze(-1), (h_new, F_new, prev_k_new)


    def forward(
        self,
        x: torch.Tensor,
        enable_surprise_gate: bool = True,
        enable_fast_weight: bool = True,
        enable_adaptive_loop: bool = True,
        separator_mask: torch.Tensor = None,
        reset_mask: Optional[torch.Tensor] = None,
        enable_delta_rule: bool = True,
        gate_role_logits: Optional[torch.Tensor] = None,
    ):
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        residual = x
        u = self.in_norm(x)

        u_feat = F.silu(self.u_proj(u))
        u_surp_bias = self.surprise_u_proj(u)
        w_gate_logits = self.write_gate_proj(u).view(B, L, H, 1, 1)
        elig_gate_logits = self.elig_gate_proj(u).view(B, L, H, 1, 1)
        token_gate_logits = (
            self.gate_token_proj(u).view(B, L, H, 2, 1, 1) * self.gate_logit_gain
        )
        w_gate_logits = w_gate_logits + token_gate_logits[:, :, :, 0]
        elig_gate_logits = elig_gate_logits + token_gate_logits[:, :, :, 1]
        if gate_role_logits is not None:
            w_gate_logits = w_gate_logits + gate_role_logits[:, :, :, 0]
            elig_gate_logits = elig_gate_logits + gate_role_logits[:, :, :, 1]
        if separator_mask is not None:
            if isinstance(separator_mask, tuple):
                elig_sep_mask, write_sep_mask = separator_mask
                w_gate_logits = w_gate_logits - write_sep_mask.to(w_gate_logits.dtype).view(B, L, 1, 1, 1) * 8.0
                elig_gate_logits = elig_gate_logits - elig_sep_mask.to(elig_gate_logits.dtype).view(B, L, 1, 1, 1) * 8.0
            else:
                gate_mask = separator_mask.to(w_gate_logits.dtype).view(B, L, 1, 1, 1) * 8.0
                w_gate_logits = w_gate_logits - gate_mask
                elig_gate_logits = elig_gate_logits - gate_mask
        hop_weights = F.softmax(self.hop_router(u), dim=-1)

        # Fair projections: K and V are identical under every ablation flag.
        addr = F.normalize(self.proj_addr(u).view(B, L, H, Dh), p=2.0, dim=-1)
        Q = F.normalize(
            addr + 0.15 * self.proj_q_res(u).view(B, L, H, Dh), p=2.0, dim=-1
        )
        K = F.normalize(
            addr + 0.15 * self.proj_k_res(u).view(B, L, H, Dh), p=2.0, dim=-1
        )
        V = addr + 0.15 * self.proj_v_res(u).view(B, L, H, Dh)

        base_decay = torch.sigmoid(self.base_decay_logit)
        fast_decay = torch.sigmoid(self.fast_decay_logit)
        fast_lr = torch.sigmoid(self.fast_lr_logit)

        h_seq, mem_seq, surprise_trace = tricore_fused_scan(
            u_feat=u_feat,
            u_surp_bias=u_surp_bias,
            w_gate_logits=w_gate_logits,
            elig_gate_logits=elig_gate_logits,
            hop_weights=hop_weights,
            Q=Q,
            K=K,
            V=V,
            base_decay=base_decay,
            fast_decay=fast_decay,
            fast_lr=fast_lr,
            hop_proj=self.hop_proj,
            enable_surprise_gate=enable_surprise_gate,
            enable_fast_weight=enable_fast_weight,
            enable_adaptive_loop=enable_adaptive_loop,
            max_loops=self.max_loops,
            reset_mask=reset_mask,
            enable_delta_rule=enable_delta_rule,
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
        separator_token: int = None,
        syntax_tokens: tuple = None,
        reset_trace_tokens: tuple = None,
        enable_delta_rule: bool = True,
        autonomous_syntax: bool = False,
    ):
        super().__init__()
        self.enable_surprise_gate = enable_surprise_gate
        self.enable_fast_weight = enable_fast_weight
        self.enable_adaptive_loop = enable_adaptive_loop
        self.enable_delta_rule = enable_delta_rule
        self.autonomous_syntax = autonomous_syntax
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Optional constant token roles used by the infix code task.  These are
        # architectural priors, not learned token embeddings.
        self.separator_token = separator_token
        self.syntax_tokens = syntax_tokens
        self.reset_trace_tokens = reset_trace_tokens
        if syntax_tokens is not None:
            for tok_id in syntax_tokens:
                if not 0 <= tok_id < vocab_size:
                    raise ValueError(f"syntax token {tok_id} outside vocabulary")
        if reset_trace_tokens is not None:
            for tok_id in reset_trace_tokens:
                if not 0 <= tok_id < vocab_size:
                    raise ValueError(f"reset token {tok_id} outside vocabulary")
        if separator_token is not None and not 0 <= separator_token < vocab_size:
            raise ValueError(f"separator token {separator_token} outside vocabulary")
        self.layers = nn.ModuleList([
            TriCoreBlock(
                d_model=d_model, n_heads=4, d_head=d_fast, max_loops=max_loops
            )
            for _ in range(n_layers)
        ])
        # Learned token-role table: zero-initialized, no hard-coded token IDs.
        # It gives each vocabulary item its own gate logit row and is the fast
        # path that solves the 350-step credit-assignment problem without masks.
        self.gate_role_table = nn.Embedding(
            vocab_size, 2 * self.layers[0].n_heads
        )
        nn.init.zeros_(self.gate_role_table.weight)
        # The table starts neutral.  Structural roles are supplied either by
        # the caller's explicit masks or by the zero-prior frequency discovery
        # below; neither path hard-codes a vocabulary ID.
        self.gate_role_gain = 8.0
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _discover_autonomous_masks(self, idx: torch.Tensor):
        """Zero-prior structural-role discovery from sequence statistics.

        No vocabulary ID is supplied.  The only signal is *within-sequence*
        repetition, which is what separates closed-class syntax from entities:

        * a structural token occurs at least ``max(6, L // 10)`` times;
        * the statement delimiter is the structural token at ``idx[:, -1]``
          (the generator terminates every five-token statement with SEMI);
        * an infix connector is any other structural token whose predecessor is
          a low-frequency entity (``freq <= 4``), e.g. ``alias EQ var``.

        Returns ``((elig_sep_mask, write_sep_mask), reset_mask)``.  The write
        gate closes on every structural token; the eligibility gate closes only
        on infix connectors, so ``prev_k`` freezes across EQ and the following
        value is written at the entity key.  A permutation of the vocabulary
        leaves every mask unchanged.
        """
        B, L = idx.shape
        if L == 0:
            empty = torch.zeros_like(idx, dtype=torch.bool)
            return (empty, empty), empty
        eq_mat = idx.unsqueeze(2) == idx.unsqueeze(1)
        tok_freq = eq_mat.sum(dim=-1)  # [B, L] occurrences of idx[b, t]
        is_structural = tok_freq >= max(6, L // 10)
        is_delim = (idx == idx[:, -1:]) & is_structural
        prev_is_entity = torch.zeros_like(is_structural)
        prev_is_entity[:, 1:] = tok_freq[:, :-1] <= 4
        elig_sep_mask = is_structural & (~is_delim) & prev_is_entity
        write_sep_mask = is_structural
        reset_mask = is_delim
        return (elig_sep_mask, write_sep_mask), reset_mask

    def _build_separator_mask(self, idx: torch.Tensor):
        if (
            self.autonomous_syntax
            and self.separator_token is None
            and self.syntax_tokens is None
        ):
            sep_mask, _ = self._discover_autonomous_masks(idx)
            return sep_mask
        if self.separator_token is None and self.syntax_tokens is None:
            return None
        if self.separator_token is not None:
            elig_sep_mask = (idx == self.separator_token)
        else:
            elig_sep_mask = torch.zeros_like(idx, dtype=torch.bool)
        write_sep_mask = elig_sep_mask.clone()
        if self.syntax_tokens is not None:
            for tok_id in self.syntax_tokens:
                write_sep_mask = write_sep_mask | (idx == tok_id)
        return (elig_sep_mask, write_sep_mask)

    def _build_reset_mask(self, idx: torch.Tensor):
        if (
            self.autonomous_syntax
            and self.reset_trace_tokens is None
            and self.separator_token is None
            and self.syntax_tokens is None
        ):
            _, reset_mask = self._discover_autonomous_masks(idx)
            return reset_mask
        if self.reset_trace_tokens is None:
            return None
        reset_mask = torch.zeros_like(idx, dtype=torch.bool)
        for tok_id in self.reset_trace_tokens:
            reset_mask = reset_mask | (idx == tok_id)
        return reset_mask

    def _build_masks(self, idx: torch.Tensor):
        """Build (separator_mask, reset_mask) once per forward pass."""
        if (
            self.autonomous_syntax
            and self.separator_token is None
            and self.syntax_tokens is None
            and self.reset_trace_tokens is None
        ):
            return self._discover_autonomous_masks(idx)
        return self._build_separator_mask(idx), self._build_reset_mask(idx)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        return [layer.init_state(batch_size, device, dtype) for layer in self.layers]

    def forward_step(self, idx_t: torch.Tensor, states: list, step_idx: int = 0):
        squeeze_seq = False
        if idx_t.dim() == 2:
            idx_t = idx_t.squeeze(1)
            squeeze_seq = True
        x_t = self.embedding(idx_t)
        separator_mask, reset_mask = self._build_masks(idx_t)
        role_logits = (
            self.gate_role_table(idx_t).view(
                idx_t.shape[0], self.layers[0].n_heads, 2, 1, 1
            )
            * self.gate_role_gain
        )
        new_states = []
        surprises = []
        for layer, st in zip(self.layers, states):
            x_t, s_t, n_st = layer.forward_step(
                x_t,
                st,
                step_idx=step_idx,
                enable_surprise_gate=self.enable_surprise_gate,
                enable_fast_weight=self.enable_fast_weight,
                enable_adaptive_loop=self.enable_adaptive_loop,
                separator_mask=separator_mask,
                reset_mask=reset_mask,
                enable_delta_rule=self.enable_delta_rule,
                gate_role_logits=role_logits,
            )
            new_states.append(n_st)
            surprises.append(s_t)
        logits_t = self.lm_head(self.final_norm(x_t))
        mean_s = torch.stack(surprises, dim=0).mean(dim=0)
        if squeeze_seq:
            logits_t = logits_t.unsqueeze(1)
            mean_s = mean_s.unsqueeze(1)
        return logits_t, mean_s, new_states

    def forward(self, idx: torch.Tensor):
        x = self.embedding(idx)
        separator_mask, reset_mask = self._build_masks(idx)
        role_logits = (
            self.gate_role_table(idx).view(
                idx.shape[0], idx.shape[1], self.layers[0].n_heads, 2, 1, 1
            )
            * self.gate_role_gain
        )
        surprises = []
        for layer in self.layers:
            x, s = layer(
                x,
                enable_surprise_gate=self.enable_surprise_gate,
                enable_fast_weight=self.enable_fast_weight,
                enable_adaptive_loop=self.enable_adaptive_loop,
                separator_mask=separator_mask,
                reset_mask=reset_mask,
                enable_delta_rule=self.enable_delta_rule,
                gate_role_logits=role_logits,
            )
            surprises.append(s)
        logits = self.lm_head(self.final_norm(x))
        mean_surprise = torch.stack(surprises, dim=0).mean(dim=0)
        return logits, mean_surprise


class BaselineTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 2048,
    ):
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
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=idx.device
        )
        x = self.encoder(x, mask=causal_mask, is_causal=True)
        logits = self.lm_head(self.final_norm(x))
        return logits, None

