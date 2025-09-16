import torch
import torch.nn.functional as F

@torch.no_grad()
def _row_normalize(x, mask=None, eps=1e-8):
    # x: (..., K); mask: (..., K) with 1 for valid keys
    if mask is not None:
        x = x * mask.float()
    z = x.sum(dim=-1, keepdim=True)
    return x / (z + eps)

@torch.no_grad()
def check_row_normalize(x, seed_q_start, seed_q_end, mask=None, eps=1e-8):
    # check row normalization in rows from seed_q_start to seed_q_end
    B, H, Lq, Lk = x.shape
    if mask is not None:
        x = x * mask.float()
    for b in range(B):
        for h in range(H):
            seed_start = int(seed_q_start[b])
            seed_end = int(seed_q_end[b])
            rowsum = x[b,h, seed_start:seed_end+1, :].sum(dim=-1, keepdim=True) # (1, 1, seed_len, 1)
            if not torch.allclose(rowsum, torch.ones_like(rowsum), atol=1e-2, rtol=1e-2):
                return False
    return True

def build_seed_diag_prior(
    Lq:int, Lk:int,
    seed_q_start: torch.Tensor,  # (B,) mRNA seed start (inclusive)
    seed_q_end:   torch.Tensor,  # (B,) mRNA seed end   (inclusive)
    k_seed_start: int = 1,       # miRNA seed start index (commonly 1 if 0-based miRNA idx; adjust to your tokenizer)
    sigma: float = 0.5,
    q_mask: torch.Tensor = None, # (B, Lq) optional mRNA attn mask (1/0)
    k_mask: torch.Tensor = None, # (B, Lk) optional miRNA attn mask (1/0)
    device=None, dtype=None,
):
    """
    Returns P of shape (B, 1, Lq, Lk): a row-normalized Gaussian diagonal prior
    centered on a linear mapping from mRNA seed positions to miRNA seed positions.
    Prior is nonzero only for mRNA rows inside [seed_q_start, seed_q_end).
    """
    B = seed_q_start.shape[0]
    device = device or seed_q_start.device
    dtype  = dtype  or torch.float32

    # indices
    q_idx = torch.arange(Lq, device=device).view(1, Lq, 1)          # (1, Lq, 1)
    k_idx = torch.arange(Lk, device=device).view(1, 1, Lk)          # (1, 1, Lk)

    # per-sample lengths
    q_s = seed_q_start.view(B, 1, 1)                                # (B,1,1)
    q_e = seed_q_end.view(B, 1, 1)
    q_len = q_e - q_s + 1                                           # (B,1,1)

    # map each mRNA seed row -> a miRNA seed center
    # let miRNA seed length match mRNA seed length 
    k_len = q_len.clone()
    k_s   = torch.full_like(q_s, int(k_seed_start))
    k_e   = k_s + k_len - 1

    # linear mapping q -> center j(q) within miRNA seed span
    # handle len==1 by zero slope
    slope = (k_len - 1).clamp_min(0) / (q_len - 1).clamp_min(1)     # (B,1,1)
    # center index in miRNA space (float)
    center = k_s + slope * (q_idx - q_s).clamp_min(0)               # (B,Lq,1)

    # Gaussian over k around 'center'
    dist2 = (k_idx - center).pow(2)                                 # (B,Lq,Lk)
    prior = torch.exp(- dist2 / (2*(sigma**2))).to(dtype)

    # zero out rows not in q-seed band
    in_q_seed = (q_idx >= q_s) & (q_idx <= q_e)                      # (B,Lq,1)
    # zero out cols not in k-seed band
    in_k_seed = (k_idx >= k_s) & (k_idx <= k_e)                      # (B,1,Lk)
    prior = prior * in_q_seed * in_k_seed

    # apply masks if provided
    if q_mask is not None:
        prior = prior * q_mask.view(B, Lq, 1)
    if k_mask is not None:
        prior = prior * k_mask.view(B, 1, Lk)

    # row-normalize per (B, q) across K
    prior = _row_normalize(prior, None)                             # (B,Lq,Lk)

    # add head dim as 1 so it can broadcast to (B,H,Lq,Lk)
    return prior.unsqueeze(1) 

def kl_diag_seed_loss(
    attn: torch.Tensor,          # (B,H,Lq,Lk)
    seed_q_start: torch.Tensor,  # (B,)
    seed_q_end: torch.Tensor,    # (B,)
    q_mask: torch.Tensor,        # (B,Lq) 1/0
    k_mask: torch.Tensor,        # (B,Lk) 1/0
    y_pos: torch.Tensor,         # (B,) 1=positive, 0=negative
    sigma: float = 1.0,
    k_seed_start: int = 1,
    eps: float = 1e-8,
):
    B, H, Lq, Lk = attn.shape
    # combine q_mask and k_mask
    q_mask = q_mask.bool()
    k_mask = k_mask.bool()
    mask = k_mask[:, None, None, :] & q_mask[:, None, :, None] # (B,1,Lq,Lk)
    mask = mask.expand(B, H, Lq, Lk) # (B,H,Lq,Lk)
    
    # Ensure attention is row-normalized across K
    if not check_row_normalize(attn, seed_q_start, seed_q_end, mask=mask, eps=eps):
        print("Row normalization failed, renormalizing over keys")
        # row normalize attn
        attn = _row_normalize(attn, mask=mask)

    # Build prior (B,1,Lq,Lk), then broadcast to H
    P = build_seed_diag_prior(
        Lq, Lk,
        seed_q_start, seed_q_end,
        k_seed_start=k_seed_start,
        sigma=sigma,
        q_mask=q_mask, k_mask=k_mask,
        device=attn.device, dtype=attn.dtype
    )                                                               # (B,1,Lq,Lk)
    P = P.expand(B, H, Lq, Lk)                                      # (B,H,Lq,Lk)

    # Seed row mask
    q_idx = torch.arange(Lq, device=attn.device).view(1, 1, Lq, 1)  # (1,1,Lq,1)
    q_s = seed_q_start.view(B,1,1,1)                                # (B,1,1,1)
    q_e = seed_q_end.view(B,1,1,1)
    q_len = q_e - q_s + 1                                            # (B,1,1,1)
    seed_row = (q_idx >= q_s) & (q_idx <= q_e)                       # (B,1,Lq,1)

    # Seed col mask
    k_idx = torch.arange(Lk, device=attn.device).view(1, 1, 1, Lk)                                          # (1,1,1,Lk)
    k_s = torch.full((B,), fill_value=k_seed_start, device=attn.device, dtype=torch.long).view(B,1,1,1)     # (B,1,1,1)
    k_e = k_s + q_len -1                                                                                        # (B,1,1,1)
    seed_col = (k_idx >= k_s) & (k_idx <= k_e)                                                               # (B,1,1,Lk)

    # Positive samples only
    pos = y_pos.view(B,1,1,1).bool()                                # (B,1,1,1)
    row_mask = seed_row & seed_col & pos                            # (B,1,Lq,Lk)
    if q_mask is not None:
        row_mask = row_mask & q_mask.view(B,1,Lq,1).bool()           # (B,1,Lq,1)
    if k_mask is not None:
        row_mask = row_mask & k_mask.view(B,1,1,Lk).bool()           # (B,1,Lq,Lk)

    # KL(A||P) = sum_k A * (log(A) - log(P))
    A = attn.clamp_min(eps)
    P = P.clamp_min(eps)
    kl = A * (A.log() - P.log())        # (B,H,Lq,Lk)

    # zero out non-seed / negatives
    kl = kl * row_mask

    # mean over valid rows/heads/batch
    return kl.sum() / row_mask.float().sum().clamp_min(1.0)