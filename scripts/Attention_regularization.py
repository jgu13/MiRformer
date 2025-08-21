import torch
import torch.nn.functional as F

@torch.no_grad()
def _row_normalize(x, mask=None, eps=1e-8):
    # x: (..., K); mask: (..., K) with 1 for valid keys
    if mask is not None:
        x = x * mask
    z = x.sum(dim=-1, keepdim=True).clamp_min(eps)
    return x / z

def build_seed_diag_prior(
    Lq:int, Lk:int,
    seed_q_start: torch.Tensor,  # (B,) mRNA seed start (inclusive)
    seed_q_end:   torch.Tensor,  # (B,) mRNA seed end   (exclusive)
    k_seed_start: int = 1,       # miRNA seed start index (commonly 1 if 0-based miRNA idx; adjust to your tokenizer)
    sigma: float = 1.0,
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
    q_len = (q_e - q_s).clamp_min(1)                                # (B,1,1)

    # map each mRNA seed row -> a miRNA seed center
    # let miRNA seed length match mRNA seed length 
    k_len = q_len.clone()
    k_s   = torch.full_like(q_s, int(k_seed_start))
    k_e   = k_s + k_len

    # linear mapping q -> center j(q) within miRNA seed span
    # handle len==1 by zero slope
    slope = (k_len - 1).clamp_min(0) / (q_len - 1).clamp_min(1)     # (B,1,1)
    # center index in miRNA space (float)
    center = k_s + slope * (q_idx - q_s).clamp_min(0)               # (B,Lq,1)

    # Gaussian over k around 'center'
    dist2 = (k_idx - center).pow(2)                                 # (B,Lq,Lk)
    prior = torch.exp(- dist2 / (2*(sigma**2))).to(dtype)

    # zero out rows not in q-seed band
    in_q_seed = (q_idx >= q_s) & (q_idx < q_e)                      # (B,Lq,1)
    prior = prior * in_q_seed

    # zero out cols not in k-seed band
    in_k_seed = (k_idx >= k_s) & (k_idx < k_e)                      # (B,1,Lk)
    prior = prior * in_k_seed

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
    attn: torch.Tensor,          # (B,H,Lq,Lk)  your row-normalized cross-attn
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
    # Ensure attention is row-normalized across K
    attn = _row_normalize(attn, k_mask[:, None, None, :].expand(B, H, Lq, Lk) if k_mask is not None else None, eps)

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
    q_idx = torch.arange(Lq, device=attn.device).view(1, 1, Lq, 1)
    q_s = seed_q_start.view(B,1,1,1)
    q_e = seed_q_end.view(B,1,1,1)
    seed_row = (q_idx >= q_s) & (q_idx < q_e)                       # (B,1,Lq,1)

    # Positive samples only
    pos = y_pos.view(B,1,1,1).bool()
    row_mask = seed_row & pos                                       # (B,1,Lq,1)
    if q_mask is not None:
        row_mask = row_mask & q_mask.view(B,1,Lq,1).bool()

    # KL(A||P) = sum_k A * (log(A) - log(P))
    A = attn.clamp_min(eps)
    P = P.clamp_min(eps)
    kl = (A * (A.log() - P.log())).sum(dim=-1, keepdim=True)        # (B,H,Lq,1)

    # zero out non-seed / negatives
    kl = kl * row_mask

    # mean over valid rows/heads/batch
    denom = row_mask.float().sum().clamp_min(1.0)
    return kl.sum() / denom