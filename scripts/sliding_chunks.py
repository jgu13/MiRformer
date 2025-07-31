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


def sliding_chunks_matmul_qk(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    '''Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size w'''
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % (w * 2) == 0
    assert q.size() == k.size()

    chunks_count = seqlen // w - 1

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    chunk_q = _chunk(q, w)
    chunk_k = _chunk(k, w)

    # matrix multipication
    # bcxd: bsz*num_heads x chunks x 2w x head_dim
    # bcyd: bsz*num_heads x chunks x 2w x head_dim
    # bcxy: bsz*num_heads x chunks x 2w x 2w
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply

    # convert diagonals into columns
    diagonal_chunk_attn = _skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)

    # allocate space for the overall attention matrix where the chunks are compined. The last dimension
    # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
    # w previous words). The following column is attention score from each word to itself, then
    # followed by w columns for the upper triangle.

    diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))

    # copy parts from diagonal_chunk_attn into the compined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    # - copying the lower triangle
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]

    # separate bsz and num_heads dimensions again
    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)

    mask_invalid_locations(diagonal_attn, w, 1, False)
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

def sliding_window_cross_attention(Q, K, V, w, mask=None):
    """
    Q: (B, Lq, H, D)
    K: (B, Lk, H, D)
    V: (B, lk, H, D)
    w: window size
    返回 attn_scores: (B, H, Lq, Lk)
    """
    B, H, Lq, D = Q.shape
    _, _, Lk, _  = K.shape

    assert Lq % (2*w) == 0

    # 1) 合并 B,H 维度
    Qr = Q.reshape(B*H, Lq, D)   # (B*H, Lq, D)
    Kr = K.reshape(B*H, Lk, D)   # (B*H, Lk, D)

    # 2) chunk 只有 Q
    Qc = _chunk(Qr, w)  # (B*H, num_chunks, 2w, D)
    # 3) 把 K expand 到相同的 num_chunks
    num_chunks = Qc.shape[1]
    Kc = Kr.unsqueeze(1).expand(-1, num_chunks, -1, -1)  # (B*H, num_chunks, Lk, D)

    # 4) 对每个 chunk 做 matmul  -> (B*H, num_chunks, 2w, Lk)
    #    b    c    x   d       b    c    k    d
    scale = 1.0 / math.sqrt(D)
    attn_chunk = torch.einsum('bcxd,bckd->bcxk', (Qc, Kc)) * scale

    # 对每个 chunk 内部掩码
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(1).expand(B, H, Lq, Lk)
        mask_r = mask.reshape(B*H, Lq, Lk)
        chunk_mask = _chunk(mask_r, w) # (B*H, num_chunks, 2w, Lk)
        attn_chunk = attn_chunk.masked_fill(chunk_mask==0, value=float('-inf'))
    attn_chunk = F.softmax(attn_chunk, dim=-1)
    
    # chunk value and weight each chunk by corresponding attn
    Vr = V.reshape(B*H, Lk, D)
    Vc = Vr.unsqueeze(1).expand(-1, num_chunks, -1, -1)  # (B*H, num_chunks, Lk, D)
    output = torch.einsum('bcxk,bckd->bcxd', (attn_chunk, Vc)) # (B*H, num_chunks, 2w, D)

    # unchunk to restore query length by taking the sum of two overlapping chunks
    # flatten the chunk dims
    bph, nc, two_w, d = output.shape
    M = nc * two_w
    out_flat = output.reshape(bph, M, d)  # → (B*H, M, D)

    # compute target indices for each chunk‐element:
    #   for chunk c in [0..nc), within‐chunk idx i in [0..2w):
    #     global position = c*w + i
    idx = (torch.arange(nc, device=output.device).unsqueeze(1) * w
           + torch.arange(two_w, device=output.device).unsqueeze(0))  # (nc,2w)
    idx = idx.reshape(-1)                                          # (M,)

    # build a [B*H, M, D] index tensor
    index = idx.unsqueeze(0).unsqueeze(-1).expand(bph, M, d)       # (B*H, M, D)

    # scatter‐add into a zero‐tensor of length Lq
    unchunk = out_flat.new_zeros(bph, Lq, d)
    unchunk.scatter_add_(1, index, out_flat)                      # (B*H, Lq, D)

    # reshape back to (B, H, Lq, D)
    unchunk = unchunk.view(B, H, Lq, d)
    return unchunk
    