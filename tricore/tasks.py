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
    2-Hop Chained Associative Recall with Adversarial Distractor Gap:
      - Hop-1 Facts (shuffled): [k_i, m_i, 0]  (binds Key k_i -> Bridge m_i)
      - Hop-2 Facts (shuffled): [m_i, v_i, 0]  (binds Bridge m_i -> Target Value v_i)
      - Adversarial Distractor Gap (length = noise_len): contains [m_i, 0, k_i, 0, ...]
        which wipes out memory if Engine I (Surprise Gate) does not close on noise token 0!
      - Queries: [k_q, 0] -> Model must predict v_q at the position of k_q in a single forward step!
    """
    n_chains = num_kv_pairs
    tier_size = (vocab_size - 4) // 3  # e.g., 20 tokens each for Keys, Bridges, Values
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

        # 1. Write Hop-1 pairs: [k_i, m_i, 0] in random order
        perm1 = torch.randperm(n_chains)
        for idx_pos, c_idx in enumerate(perm1):
            base = idx_pos * 3
            inputs[b, base] = k_toks[c_idx]
            inputs[b, base + 1] = m_toks[c_idx]

        # 2. Write Hop-2 pairs: [m_i, v_i, 0] in independent random order
        perm2 = torch.randperm(n_chains)
        for idx_pos, c_idx in enumerate(perm2):
            base = hop1_len + idx_pos * 3
            inputs[b, base] = m_toks[c_idx]
            inputs[b, base + 1] = v_toks[c_idx]

        # 3. Adversarial Distractor Gap: insert unpaired [m_i, 0] and [k_i, 0] distractors
        gap_start = hop1_len + hop2_len
        distractors = torch.cat([m_toks, k_toks])
        for d_idx in range(min(len(distractors), noise_len // 2)):
            inputs[b, gap_start + d_idx * 2] = distractors[d_idx]
            # Position gap_start + d_idx * 2 + 1 remains 0 (noise_token_id)

        # 4. Multi-Hop Queries: present k_q, require predicting v_q (2 hops away!)
        q_offset = gap_start + noise_len
        q_indices = torch.randperm(n_chains)[:num_queries]
        for j, c_idx in enumerate(q_indices):
            q_pos = q_offset + j * 2
            inputs[b, q_pos] = k_toks[c_idx]
            targets[b, q_pos] = v_toks[c_idx]

    return inputs.to(device), targets.to(device)


# Contextual Code LM task tokens. Token 7 is reserved so the special-token block
# can grow without shifting the variable/alias/literal pools.
CODE_NOISE = 0
CODE_LET = 1
CODE_ALIAS = 2
CODE_EQ = 3
CODE_SEMI = 4
CODE_NOP = 5
CODE_RET = 6
CODE_VAR_BASE = 8


def generate_contextual_code_lm_batch(
    batch_size: int = 32,
    num_vars: int = 6,
    num_noise: int = 6,
    vocab_size: int = 64,
    device: str = "cpu",
):
    """
    Contextual Code LM: dense next-token prediction over a tiny imperative program.

    Each sample contains, in random statement order:
      - assignment: [LET, var_i, EQ, lit_i, SEMI]
      - alias:      [ALIAS, alias_i, EQ, var_i, SEMI]
      - filler:     [NOP, NOISE, SEMI]
    and terminates with:
      - return:     [RET, alias_i, lit_i, SEMI]

    The variable, alias, and literal namespaces are disjoint. The return uses an
    alias, so predicting `lit_i` at the alias position requires resolving
    alias_i -> var_i -> lit_i rather than copying a surface token. The function
    returns `(inputs, targets, deref_mask)`, where `targets` is the standard
    shifted next-token target (-100 at the final position) and `deref_mask`
    marks the single return-alias position used for Dereference Accuracy.
    """
    if num_vars < 1:
        raise ValueError("num_vars must be positive")
    if num_noise < 0:
        raise ValueError("num_noise must be non-negative")

    required_vocab = CODE_VAR_BASE + 3 * num_vars
    if vocab_size < required_vocab:
        raise ValueError(
            f"vocab_size={vocab_size} is too small for num_vars={num_vars}; "
            f"need at least {required_vocab}"
        )

    var_pool = CODE_VAR_BASE
    alias_pool = var_pool + num_vars
    lit_pool = alias_pool + num_vars

    statements_per_var = 2
    tokens_per_var = statements_per_var * 5
    tokens_per_noise = 3
    seq_len = num_vars * tokens_per_var + num_noise * tokens_per_noise + 4

    inputs = torch.full((batch_size, seq_len), CODE_NOISE, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    deref_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    for b in range(batch_size):
        var_ids = var_pool + torch.randperm(num_vars)
        alias_ids = alias_pool + torch.randperm(num_vars)
        lit_ids = lit_pool + torch.randperm(num_vars)

        statements = []
        for i in range(num_vars):
            var_i = int(var_ids[i])
            alias_i = int(alias_ids[i])
            lit_i = int(lit_ids[i])
            statements.append([CODE_LET, var_i, CODE_EQ, lit_i, CODE_SEMI])
            statements.append([CODE_ALIAS, alias_i, CODE_EQ, var_i, CODE_SEMI])

        for _ in range(num_noise):
            statements.append([CODE_NOP, CODE_NOISE, CODE_SEMI])

        order = torch.randperm(len(statements)).tolist()
        seq = []
        for idx in order:
            seq.extend(statements[idx])

        ret_idx = int(torch.randint(num_vars, (1,)).item())
        seq.extend([CODE_RET, int(alias_ids[ret_idx]), int(lit_ids[ret_idx]), CODE_SEMI])

        flat = torch.tensor(seq, dtype=torch.long)
        inputs[b, :seq_len] = flat
        targets[b, : seq_len - 1] = flat[1:]

        # At the return alias position, the next token is the literal value.
        deref_mask[b, seq_len - 3] = True

    return inputs.to(device), targets.to(device), deref_mask.to(device)
