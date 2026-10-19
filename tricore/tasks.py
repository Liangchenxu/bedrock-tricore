import torch


def generate_associative_recall_batch(
    batch_size: int = 32,
    num_kv_pairs: int = 8,
    num_queries: int = 4,
    vocab_size: int = 64,
    noise_len: int = 16,
    device: str = "cpu",
):
    """
    Multi-Query Associative Recall (MQAR) & Noise Resistance Task.
    Format:
      [k_1, v_1, k_2, v_2, ..., k_n, v_n] + [noise tokens...] + [q_1, ans_1, q_2, ans_2...]
    Training uses next-token prediction at each query-key position: the model
    sees q_j and must output the value token that follows. Only those query
    positions contribute to CrossEntropy loss; all other targets are -100.
    """
    half_vocab = (vocab_size - 2) // 2
    key_tokens = torch.arange(1, 1 + half_vocab)
    val_tokens = torch.arange(1 + half_vocab, 1 + 2 * half_vocab)
    noise_token_id = 0

    seq_len = num_kv_pairs * 2 + noise_len + num_queries * 2
    inputs = torch.full((batch_size, seq_len), noise_token_id, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        perm_k = key_tokens[torch.randperm(half_vocab)[:num_kv_pairs]]
        perm_v = val_tokens[torch.randperm(half_vocab)[:num_kv_pairs]]

        # Populate Key-Value context pairs
        for i in range(num_kv_pairs):
            inputs[b, i * 2] = perm_k[i]
            inputs[b, i * 2 + 1] = perm_v[i]

        # Insert distracting noise gap in the middle
        offset = num_kv_pairs * 2 + noise_len

        # Select query keys from the presented pairs
        q_indices = torch.randperm(num_kv_pairs)[:num_queries]
        for j, idx in enumerate(q_indices):
            q_pos = offset + j * 2
            inputs[b, q_pos] = perm_k[idx]
            if q_pos + 1 < seq_len:
                inputs[b, q_pos + 1] = perm_v[idx]
            # The model sees perm_k[idx] at q_pos and must output perm_v[idx] there.
            targets[b, q_pos] = perm_v[idx]

    return inputs.to(device), targets.to(device)
