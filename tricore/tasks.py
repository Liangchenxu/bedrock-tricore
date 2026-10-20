import torch


def generate_associative_recall_batch(
    batch_size: int = 64,
    num_kv_pairs: int = 6,
    num_queries: int = 4,
    vocab_size: int = 64,
    noise_len: int = 16,
    device: str = "cpu",
):
    """
    Track 1: 2-Hop Chained Associative Recall with Adversarial Distractor Gap.
      - Hop-1 Facts (shuffled): [k_i, m_i, 0]  (binds Key k_i -> Bridge m_i)
      - Hop-2 Facts (shuffled): [m_i, v_i, 0]  (binds Bridge m_i -> Target Value v_i)
      - Adversarial Distractor Gap (length = noise_len): repeated [m_i, 0, k_i, 0] overwrites
        that corrupt working memory unless Engine I (Surprise/Write Gate) blocks delimiter token 0.
      - Queries: [k_q, 0] -> Predict v_q at the position of k_q in a single forward step!
    """
    n_chains = num_kv_pairs
    tier_size = (vocab_size - 4) // 3
    key_pool = torch.arange(1, 1 + tier_size)
    bridge_pool = torch.arange(1 + tier_size, 1 + 2 * tier_size)
    val_pool = torch.arange(1 + 2 * tier_size, 1 + 3 * tier_size)
    noise_token_id = 0

    hop1_len = n_chains * 3
    hop2_len = n_chains * 3
    query_len = num_queries * 2
    seq_len = hop1_len + hop2_len + noise_len + query_len

    inputs = torch.full((batch_size, seq_len), noise_token_id, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        k_toks = key_pool[torch.randperm(tier_size)[:n_chains]]
        m_toks = bridge_pool[torch.randperm(tier_size)[:n_chains]]
        v_toks = val_pool[torch.randperm(tier_size)[:n_chains]]

        perm1 = torch.randperm(n_chains)
        for idx_pos, c_idx in enumerate(perm1):
            base = idx_pos * 3
            inputs[b, base] = k_toks[c_idx]
            inputs[b, base + 1] = m_toks[c_idx]

        perm2 = torch.randperm(n_chains)
        for idx_pos, c_idx in enumerate(perm2):
            base = hop1_len + idx_pos * 3
            inputs[b, base] = m_toks[c_idx]
            inputs[b, base + 1] = v_toks[c_idx]

        # Adversarial Distractor Gap: cycle through all bridges and keys paired with noise token 0
        gap_start = hop1_len + hop2_len
        distractors = torch.cat([m_toks, k_toks, m_toks, k_toks])
        num_dist_pairs = noise_len // 2
        for d_idx in range(num_dist_pairs):
            tok = distractors[d_idx % len(distractors)]
            inputs[b, gap_start + d_idx * 2] = tok
            inputs[b, gap_start + d_idx * 2 + 1] = noise_token_id

        q_offset = gap_start + noise_len
        q_indices = torch.randperm(n_chains)[:num_queries]
        for j, c_idx in enumerate(q_indices):
            q_pos = q_offset + j * 2
            inputs[b, q_pos] = k_toks[c_idx]
            targets[b, q_pos] = v_toks[c_idx]

    return inputs.to(device), targets.to(device)


def generate_contextual_code_lm_batch(
    batch_size: int = 32,
    num_vars: int = 6,
    num_queries: int = 4,
    vocab_size: int = 64,
    num_nops: int = 4,
    device: str = "cpu",
):
    """
    Track 2: Contextual Code Language Modeling & Pointer Dereferencing (Causal Bytecode Format).
    Tokens:
      0: SEMI (;), 60: LET, 61: ALIAS, 62: NOP, 63: RET
      1..19: alias tokens (k_i)
      20..38: variable pointer tokens (m_i)
      39..57: literal value tokens (v_i)
    Bytecode Statements:
      - Declaration: [LET, alias_i, var_i, SEMI]      (binds alias_i -> var_i)
      - Assignment:  [ALIAS, var_i, lit_i, SEMI]      (binds var_i -> lit_i)
      - Distractor:  [NOP, var_i, SEMI, SEMI]         (adversarial no-op statement)
      - Return Eval: [RET, alias_q, lit_q, SEMI]      (predicts lit_q right after alias_q!)
    Returns:
      inputs: [B, L]
      lm_targets: [B, L] (dense next-token targets for Perplexity evaluation)
      deref_mask: [B, L] (boolean mask marking the exact return dereference positions)
    """
    SEMI, LET, ALIAS, NOP, RET = 0, 60, 61, 62, 63
    tier_size = 18
    alias_pool = torch.arange(1, 1 + tier_size)
    var_pool = torch.arange(1 + tier_size, 1 + 2 * tier_size)
    lit_pool = torch.arange(1 + 2 * tier_size, 1 + 3 * tier_size)

    total_stmts = num_vars * 2 + num_nops + num_queries
    seq_len = total_stmts * 4

    seq = torch.full((batch_size, seq_len + 1), SEMI, dtype=torch.long)
    deref_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    for b in range(batch_size):
        a_toks = alias_pool[torch.randperm(tier_size)[:num_vars]]
        v_toks = var_pool[torch.randperm(tier_size)[:num_vars]]
        l_toks = lit_pool[torch.randperm(tier_size)[:num_vars]]

        pos = 0
        # 1. Emit shuffled [LET, alias_i, var_i, SEMI]
        for idx in torch.randperm(num_vars):
            seq[b, pos : pos + 4] = torch.tensor([LET, a_toks[idx], v_toks[idx], SEMI])
            pos += 4

        # 2. Emit shuffled [ALIAS, var_i, lit_i, SEMI]
        for idx in torch.randperm(num_vars):
            seq[b, pos : pos + 4] = torch.tensor([ALIAS, v_toks[idx], l_toks[idx], SEMI])
            pos += 4

        # 3. Emit distracting [NOP, var_i, SEMI, SEMI] statements
        for n_i in range(num_nops):
            v_dist = v_toks[n_i % num_vars]
            seq[b, pos : pos + 4] = torch.tensor([NOP, v_dist, SEMI, SEMI])
            pos += 4

        # 4. Emit Return evaluations [RET, alias_q, lit_q, SEMI]
        q_indices = torch.randperm(num_vars)[:num_queries]
        for c_idx in q_indices:
            seq[b, pos : pos + 4] = torch.tensor([RET, a_toks[c_idx], l_toks[c_idx], SEMI])
            # At pos + 1 (where input is alias_q), the next token is lit_q!
            deref_mask[b, pos + 1] = True
            pos += 4

    inputs = seq[:, :-1].to(device)
    lm_targets = seq[:, 1:].to(device)
    return inputs, lm_targets, deref_mask.to(device)
