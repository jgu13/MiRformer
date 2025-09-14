import torch
import torch.nn.functional as F
import math
from diagonaled_mm_tvm import mask_invalid_locations


def _skew(x, direction, padding_value):
    '''Convert diagonals into columns (or columns into diagonals depending on `direction`'''
    x_padded = F.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
    return x_padded


def _skew2(x, padding_value):
    '''shift every row 1 step to right converting columns into diagonals'''
    # X = B x C x M x L
    B, C, M, L = x.size()
    x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
    x = x.view(B, C, -1)  # B x C x ML+MM+M
    x = x[:, :, :-M]  # B x C x ML+MM
    x = x.view(B, C, M, M + L)  # B x C, M x L+M
    x = x[:, :, :, :-1]
    return x


def _chunk(x, w):
    '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''

    # non-overlapping chunks of size = 2w
    x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

    # use `as_strided` to make the chunks overlap with an overlap size = w
    chunk_size = list(x.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(x.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return x.as_strided(size=chunk_size, stride=chunk_stride)


def sliding_chunks_matmul_qk(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float, eps=1e-8):
    '''Matrix multiplication of query x key tensors using a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size w. Enhanced with numerical stability improvements.
    
    Args:
        q: Query tensor (bsz, seqlen, num_heads, head_dim)
        k: Key tensor (bsz, seqlen, num_heads, head_dim)
        w: Window size (one-sided)
        padding_value: Value to use for padding
        eps: Small epsilon for numerical stability
    '''
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % (w * 2) == 0, f"Sequence length {seqlen} must be divisible by {2*w}"
    assert q.size() == k.size(), f"Query and key must have same size: {q.size()} vs {k.size()}"
    assert head_dim > 0, "Head dimension must be positive"

    # Check for NaN/Inf in inputs
    if torch.isnan(q).any() or torch.isinf(q).any():
        raise ValueError("Query tensor contains NaN or Inf values")
    if torch.isnan(k).any() or torch.isinf(k).any():
        raise ValueError("Key tensor contains NaN or Inf values")

    chunks_count = seqlen // w - 1

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    chunk_q = _chunk(q, w)
    chunk_k = _chunk(k, w)

    # matrix multiplication with improved stability
    # bcxd: bsz*num_heads x chunks x 2w x head_dim
    # bcyd: bsz*num_heads x chunks x 2w x head_dim
    # bcxy: bsz*num_heads x chunks x 2w x 2w
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply

    # Clip attention scores to prevent overflow
    chunk_attn = torch.clamp(chunk_attn, min=-1e6, max=1e6)

    # convert diagonals into columns
    diagonal_chunk_attn = _skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)

    # allocate space for the overall attention matrix where the chunks are combined. The last dimension
    # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
    # w previous words). The following column is attention score from each word to itself, then
    # followed by w columns for the upper triangle.

    diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))

    # copy parts from diagonal_chunk_attn into the combined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    # - copying the lower triangle
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]

    # separate bsz and num_heads dimensions again
    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)

    # Apply masking for invalid locations
    mask_invalid_locations(diagonal_attn, w, 1, False)
    
    # Final stability check
    if torch.isnan(diagonal_attn).any() or torch.isinf(diagonal_attn).any():
        print("Warning: Attention scores contain NaN/Inf, applying fallback")
        diagonal_attn = torch.where(
            torch.isnan(diagonal_attn) | torch.isinf(diagonal_attn),
            torch.zeros_like(diagonal_attn),
            diagonal_attn
        )
    
    return diagonal_attn


def sliding_chunks_matmul_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
    '''Same as sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
    format from sliding_chunks_matmul_qk'''
    bsz, seqlen, num_heads, head_dim = v.size()
    assert seqlen % (w * 2) == 0
    assert prob.size()[:3] == v.size()[:3]
    assert prob.size(3) == 2 * w + 1
    chunks_count = seqlen // w - 1
    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
    chunk_prob = prob.transpose(1, 2).reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)

    # group bsz and num_heads dimensions into one
    v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    # pad seqlen with w at the beginning of the sequence and another w at the end
    padded_v = F.pad(v, (0, 0, w, w), value=-1)

    # chunk padded_v into chunks of size 3w and an overlap of size w
    chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
    chunk_v_stride = padded_v.stride()
    chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
    chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

    skewed_prob = _skew2(chunk_prob, padding_value=0)

    context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
    return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)


def pad_to_window_size(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = int(2 * one_sided_window_size)
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask


# ========= "sliding_chunks_no_overlap": alternative implemenation of the sliding window attention =========
# This implementation uses non-overlapping chunks (or blocks) of size `w` with number of local attention = 3xw
# To make this implemenation comparable to "sliding_chunks" set w such that
#       w_of_sliding_chunks_no_overlap = w_of_sliding_chunks * 2 / 3
# For example,
#    w_of_sliding_chunks = 256 (this is one sided. Total attention size = 512)
#    w_of_sliding_chunks_no_overlap = 170 (Total attention size = 510)
# Performance:
# - Speed: 30% faster than "sliding_chunks"
# - Memory: 95% of the memory usage of "sliding_chunks"
# The windows are asymmetric where number of attention on each side of a token ranges between w to 2w
# while "sliding_chunks" has a symmetric window around each token.


def sliding_chunks_no_overlap_matmul_qk(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % w == 0
    assert q.size() == k.size()
    # chunk seqlen into non-overlapping chunks of size w
    chunk_q = q.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_k = k.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_k_expanded = torch.stack((
        F.pad(chunk_k[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0),
        chunk_k,
        F.pad(chunk_k[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.0),
    ), dim=-1)
    diagonal_attn = torch.einsum('bcxhd,bcyhde->bcxhey', (chunk_q, chunk_k_expanded))  # multiply
    return diagonal_attn.reshape(bsz, seqlen, num_heads, 3 * w)


def sliding_chunks_no_overlap_matmul_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
    bsz, seqlen, num_heads, head_dim = v.size()
    chunk_prob = prob.view(bsz, seqlen // w, w, num_heads, 3, w)
    chunk_v = v.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_v_extended = torch.stack((
        F.pad(chunk_v[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0),
        chunk_v,
        F.pad(chunk_v[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.0),
    ), dim=-1)
    context = torch.einsum('bcwhpd,bcdhep->bcwhe', (chunk_prob, chunk_v_extended))
    return context.reshape(bsz, seqlen, num_heads, head_dim)

def check_key_mask_rows(key_mask, name="key_mask"):
    """
    key_mask: (B, Lk), 1=valid, 0=pad (or bool)
    Prints indices of samples that have no valid keys at all.
    """
    km = key_mask
    if km.dtype != torch.bool:
        km = km != 0  # cast to bool

    empty = ~km.any(dim=1)   # (B,), True if that sample has all zeros
    if empty.any():
        bad_idx = empty.nonzero(as_tuple=True)[0]
        counts = km.sum(dim=1)
        raise RuntimeError(f"[{name}] Found {bad_idx.numel()} samples with ALL-ZERO key mask. "
              f"indices={bad_idx.tolist()}, valid_counts={counts[bad_idx].tolist()}")

def _sum_unchunk(x, w, B, H, Lq):
    # unchunk to restore query length by taking the sum of two overlapping chunks
    # flatten the chunk dims
    bph, nc, two_w, d = x.shape
    M = nc * two_w
    out_flat = x.reshape(bph, M, d)  # → (B*H, M, D)

    # compute target indices for each chunk‐element:
    #   for chunk c in [0..nc), within‐chunk idx i in [0..2w):
    #     global position = c*w + i
    idx = (torch.arange(nc, device=x.device).unsqueeze(1) * w
           + torch.arange(two_w, device=x.device).unsqueeze(0))  # (nc,2w)
    idx = idx.reshape(-1)                                        # (M,)

    # build a [B*H, M, D] index tensor
    index = idx.unsqueeze(0).unsqueeze(-1).expand(bph, M, d)      # (B*H, M, D)

    # scatter‐add into a zero‐tensor of length Lq
    unchunk = out_flat.new_zeros(bph, Lq, d)
    unchunk.scatter_add_(1, index, out_flat)  
    
    # (B*H, Lq, D)

    # reshape back to (B, H, Lq, D)
    unchunk = unchunk.view(B, H, Lq, d)
    return unchunk

def _unchunk(x, w, B, H, Lq, Lk, slide_on_query=True, reduce='mean'):
    if slide_on_query:
        bsh, C, two_w, Lk = x.shape
        assert Lq == (C + 1) * w, f"Lq={Lq}, expected {(C+1)*w}"
    else:
        bsh, Lq, C, two_w = x.shape
        assert Lk == (C + 1) * w, f"Lk={Lk}, expected {(C+1)*w}"

    # build indices over which to take max
    # chunk index: [0,...,C], one chunk jumps w steps to the next chunk
    # Each chunk is 2w wide. Index inside each chunk: u = [0,...,2w] 
    # global index = (C * w) + u
    device = x.device
    base = torch.arange(C, device=device) * w #(C, )
    offset = torch.arange(two_w, device=device) #(2w, )
    positions = base[:, None] + offset[None, :] #(C, 2w)
    pos_flat = positions.reshape(-1) # flatten positions to (C * 2w)

    # flatten chunks in input
    if slide_on_query:
        chunks_in = x.reshape(bsh, C*two_w, Lk) # (bsh, C*2w, Lk)
    else:
        chunks_in = x.reshape(bsh, Lq, C*two_w) # (bsh, Lq, C*2w)

    # build output tensor
    if reduce == 'sum' or reduce == 'mean':
        output = torch.zeros((bsh, Lq, Lk), device=x.device, dtype=x.dtype) # so that values do not reduce to -inf
    elif reduce == 'amax':
        output = torch.full((bsh, Lq, Lk), fill_value=float('-inf'), device=device, dtype=chunks_in.dtype) # (bsh, (C+1)*w, d)
    if slide_on_query:
        indices = pos_flat[None, :, None].expand(bsh, -1, Lk) # (bsh, C*2w, Lk)
    else: 
        indices = pos_flat[None, None, :].expand(bsh, Lq, -1) # (bsh, Lq, C*2w)

    # take max between values that overlaps at the same position
    if slide_on_query:
        chunk_out = output.scatter_reduce_(dim=1,
                                            index=indices,
                                            src=chunks_in,
                                            reduce=reduce,
                                            include_self=False)
    else:
        chunk_out = output.scatter_reduce_(dim=2,
                                            index=indices,
                                            src=chunks_in,
                                            reduce=reduce,
                                            include_self=False)
    chunk_out = chunk_out.view(B, H, Lq, Lk)
    return chunk_out

def _unchunk_lse_logits(logits_chunk, w, B, H, Lq, Lk, slide_on_query=True, eps=1e-10):
    """
    logits_chunk: (B*H, C, 2w, Lk)  -- per-chunk attention *logits* (NOT softmaxed)
    returns:      (B, H, Lq, Lk)    -- merged logits via log-sum-exp over overlaps

    If average=True, returns log-mean-exp (LSE - log(count)).
    """
    if slide_on_query:
        bsh, C, two_w, Lk = logits_chunk.shape
        assert Lq == (C + 1) * w, f"Lq={Lq}, expected {(C+1)*w}"
    else:
        bsh, Lq, C, two_w = logits_chunk.shape
        assert Lk == (C + 1) * w, f"Lk={Lk}, expected {(C+1)*w}" 

    device = logits_chunk.device
    # Map chunk indices (c,u) -> global query index i = c*w + u
    base   = torch.arange(C, device=device) * w                 # (C,)
    offset = torch.arange(two_w, device=device)                 # (2w,)
    pos    = base[:, None] + offset[None, :]                    # (C, 2w)
    idx    = pos.reshape(-1)                                    # (C*2w,)

    if slide_on_query:
        flat = logits_chunk.reshape(bsh, C*two_w, Lk)           # (bsh, C*2w, Lk)
    else:
        flat = logits_chunk.reshape(bsh, Lq, C*two_w)           # (bsh, Lq, C*2w)

    # 1) max per (bsh, i, k) for numerical stability
    m = torch.full((bsh, Lq, Lk), -float('inf'), device=device, dtype=flat.dtype)
    if slide_on_query:
        m.scatter_reduce_(1, idx[None, :, None].expand(bsh, -1, Lk),
                        flat, reduce='amax', include_self=True)
    else:
        m.scatter_reduce_(2, idx[None, None, :].expand(bsh, Lq, -1),
                        flat, reduce='amax', include_self=True)

    # 2) sum exp(logit - m) across overlapping contributions
    exp_acc = torch.zeros_like(m)
    # gather m back to the flat layout to align
    if slide_on_query:
        m_flat = m.gather(1, idx[None, :, None].expand(bsh, -1, Lk))
        exp_acc.scatter_add_(1, idx[None, :, None].expand(bsh, -1, Lk), torch.exp(flat - m_flat))
    else:
        m_flat = m.gather(2, idx[None, None, :].expand(bsh, Lq, -1))
        exp_acc.scatter_add_(2, idx[None, None, :].expand(bsh, Lq, -1), torch.exp(flat - m_flat))

    # 3) LSE = m + log(sum_exp)
    merged = m + torch.log(exp_acc + eps)
    return merged.view(B,H,Lq,Lk)


def sliding_window_cross_attention(Q, K, V, w, mask=None, norm_by_query=False, 
                                  eps=1e-8, max_attn_value=1e6, use_lse=False):
    """
    Enhanced sliding window cross-attention with improved numerical stability.
    
    Args:
        Q: (B, H, Lq, D) - Query tensor
        K: (B, H, Lk, D) - Key tensor  
        V: (B, H, Lk, D) - Value tensor
        w: window size
        mask: attention mask (B, Lq, Lk) where 1 = attend, 0 = mask
        norm_by_query: whether to normalize attention by query dimension
        eps: small epsilon for numerical stability
        max_attn_value: maximum attention value to prevent overflow
        clip_grad_norm: gradient clipping norm (if None, no clipping)
    
    Returns:
        (output, attention_weights) where:
        - output: (B, H, Lq, D) - weighted value output
        - attention_weights: (B, H, Lq, Lk) - attention probabilities
    """
    B, H, Lq, D = Q.shape
    _, _, Lk, _ = K.shape

    slide_on_query = True
    if Lk > Lq:
        slide_on_query = False

    if slide_on_query:
        assert Lq % (2*w) == 0, f"Lq ({Lq}) must be divisible by 2*w ({2*w})"
    else:
        assert Lk % (2*w) == 0, f"Lk ({Lk}) must be divisible by 2*w ({2*w})"
    assert D > 0, "Head dimension must be positive"
    
    # Check for NaN/Inf in inputs
    if torch.isnan(Q).any() or torch.isinf(Q).any():
        raise ValueError("Query tensor contains NaN or Inf values")
    if torch.isnan(K).any() or torch.isinf(K).any():
        raise ValueError("Key tensor contains NaN or Inf values")
    if torch.isnan(V).any() or torch.isinf(V).any():
        raise ValueError("Value tensor contains NaN or Inf values")

    # 1) Merge B,H dimensions for efficient processing
    Qr = Q.reshape(B*H, Lq, D)   # (B*H, Lq, D)
    Kr = K.reshape(B*H, Lk, D)   # (B*H, Lk, D)

    if slide_on_query:
        # 2) Chunk only Q for sliding window
        Qc = _chunk(Qr, w)  # (B*H, num_chunks, 2w, D)
        num_chunks = Qc.shape[1]
        # 3) Expand K to match chunk structure
        Kc = Kr.unsqueeze(1).expand(-1, num_chunks, -1, -1)  # (B*H, num_chunks, Lk, D)

    else:
        # 2) Chunk only K for sliding window
        Kc = _chunk(Kr, w) # (B*H, num_chunks, 2w, D)
        num_chunks = Kc.shape[1] 
        # 3) Expand Q to match chunk structure
        Qc = Qr.unsqueeze(1).expand(-1, num_chunks, -1, -1)  # (B*H, num_chunks, Lq, D)

    # 4) Compute attention scores with improved numerical stability
    # Use more stable scaling
    scale = 1.0 / math.sqrt(D)
    
    if slide_on_query:
        # Compute attention scores
        attn_chunk = torch.einsum('bcxd,bckd->bcxk', (Qc, Kc)) * scale # (B*H, num_chunks, 2w, Lk)
    else:
        # Compute attention scores
        attn_chunk = torch.einsum('bcxd,bckd->bcxk', (Qc, Kc)) * scale # (B*H, num_chunks, Lq, 2w)
        attn_chunk = attn_chunk.permute(0, 2, 1, 3).contiguous() # (B*H, Lq, num_chunks, 2w)
    
    # Clip attention scores to prevent overflow
    attn_chunk = torch.clamp(attn_chunk, min=-max_attn_value, max=max_attn_value)

    # Apply attention mask if provided
    if mask is not None:
        # assert mask is bool
        assert mask.dtype == torch.bool, f"mask dtype is {mask.dtype}, expected bool"
        # Expand mask to match attention dimensions (B,H,Lq,Lk)
        if mask.dim() == 2:           # (B, Lk) key mask
            assert mask.shape == (B, Lk), f"mask shape is {mask.shape}, expected (B, Lk)"
            mask = mask[:, None, None, :].expand(B, H, Lq, Lk)
        elif mask.dim() == 3:         # (B, Lq, Lk)
            assert mask.shape == (B, Lq, Lk), f"mask shape is {mask.shape}, expected (B, Lq, Lk)"
            mask = mask[:, None, :, :].expand(B, H, Lq, Lk)
        elif mask.dim() == 4:         # (B, H, Lq, Lk)
            assert mask.shape == (B, H, Lq, Lk), f"mask shape is {mask.shape}, expected (B, H, Lq, Lk)"
        else:
            raise ValueError(f"mask shape is {mask.shape}, expected (B, Lk), (B, Lq, Lk), or (B, H, Lq, Lk)")

        if slide_on_query:
            chunk_mask = _chunk(mask, w)  # (B*H, num_chunks, 2w, Lk)
        else:
            mask = mask.transpose(1, 2).contiguous() # (B*H, Lk, Lq)
            chunk_mask = _chunk(mask, w) # (B*H, num_chunks, 2w, Lq)
            chunk_mask = chunk_mask.permute(0, 3, 1, 2).contiguous() # (B*H, Lq, num_chunks, 2w)
        
        # Apply mask with large negative value (but not too large to prevent overflow)
        mask_value = -10000.0  
        attn_chunk = attn_chunk.masked_fill(~chunk_mask, value=mask_value) # (B*H, num_chunks, 2w, Lk) if slide on query else (B*H, Lq, num_chunks, 2w)

    if use_lse:
        # unchunk by LSE
        unchunk_attn = _unchunk_lse_logits(attn_chunk, w=w, B=B, H=H, Lq=Lq, Lk=Lk, slide_on_query=slide_on_query) # (B, H, Lq, Lk)
    else:
        # fallback to mean unchunk
        unchunk_attn = _unchunk(attn_chunk, w=w, B=B, H=H, Lq=Lq, Lk=Lk, reduce="mean", slide_on_query=slide_on_query)  # (B, H, Lq, Lk)
    
    # print("logits stats:", unchunk_attn.min().item(), unchunk_attn.max().item())

    # Apply softmax 
    if norm_by_query:
        attn_weights = _stable_softmax(unchunk_attn, dim=-2, eps=eps) # (B, H, Lq, Lk)
    else:
        # Enhanced numerical stability for softmax
        attn_weights = _stable_softmax(unchunk_attn, dim=-1, eps=eps) # (B, H, Lq, Lk)
    # check all zero rows
    row_sums = attn_weights.sum(dim=-1)
    # print("row_sums min/max:", row_sums.min().item(), row_sums.max().item())
    
    # Apply attention weights to values
    output = torch.einsum('bcxk,bckd->bcxd', (attn_weights, V))  # (B, H, Lq, Lk) * (B, H, Lk, D) -> (B, H, Lq, D)
    
    return (output, attn_weights)


def _stable_softmax(x, dim=-1, eps=1e-8):
    """
    Numerically stable softmax implementation.
    
    Args:
        x: input tensor
        dim: dimension to apply softmax
        eps: small epsilon for numerical stability
    
    Returns:
        softmax output
    """
    # Subtract max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    
    # Handle case where max is -inf
    if torch.isinf(x_max).any():
        x_max = torch.where(torch.isinf(x_max), torch.zeros_like(x_max), x_max)
    
    # Compute exp(x - x_max)
    z = x - x_max
    exp_z = torch.exp(z).clamp(max=1e6)
    
    # Compute sum
    sum_exp = exp_z.sum(dim=dim, keepdim=True).clamp_min(min=eps)
    
    # Return softmax
    return exp_z / sum_exp


def _stable_log_softmax(x, dim=-1, eps=1e-8):
    """
    Numerically stable log_softmax implementation.
    
    Args:
        x: input tensor
        dim: dimension to apply log_softmax
        eps: small epsilon for numerical stability
    
    Returns:
        log_softmax output
    """
    # Subtract max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    
    # Handle case where max is -inf
    if torch.isinf(x_max).any():
        x_max = torch.where(torch.isinf(x_max), torch.zeros_like(x_max), x_max)
    
    # Compute x - x_max
    x_stable = x - x_max
    
    # Compute log sum exp
    log_sum_exp = torch.logsumexp(x_stable, dim=dim, keepdim=True)
    
    # Return log_softmax
    return x_stable - log_sum_exp 


def compute_attention_with_fallback(Q, K, V, mask=None, eps=1e-8):
    """
    Compute attention with fallback to mean pooling if numerical issues occur.
    
    Args:
        Q, K, V: Query, Key, Value tensors
        mask: Attention mask
        eps: Small epsilon for stability
    
    Returns:
        (output, attention_weights)
    """
    try:
        # Try standard attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
        
    except (RuntimeError, ValueError) as e:
        print(f"Attention computation failed: {e}. Using fallback.")
        
        # Fallback: mean pooling
        if mask is not None:
            V_masked = V * mask.unsqueeze(-1)
            output = V_masked.sum(dim=-2) / (mask.sum(dim=-1, keepdim=True) + eps)
        else:
            output = V.mean(dim=-2)
        
        # Create uniform attention weights
        attention_weights = torch.ones_like(Q[..., :1, :]) / Q.size(-2)
        
        return output, attention_weights 
    