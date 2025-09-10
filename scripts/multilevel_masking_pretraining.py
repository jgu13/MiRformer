import torch
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F

def debug_label_range(logits, labels, *, name="labels", ignore_index=-100, max_print=10):
    """
    logits : (..., V)  float
    labels : (...)     long
    """
    V = logits.size(-1)

    # force CPU for safe printing
    lab = labels.detach().to("cpu")
    bad = (lab != ignore_index) & ((lab < 0) | (lab >= V))

    if bad.any():
        idxs = bad.nonzero(as_tuple=False)  # shape [N, labels.dim()]
        vals = lab[bad]
        n = min(max_print, idxs.size(0))
        print(f"[{name}] BAD LABELS: count={idxs.size(0)}  V={V}  ignore_index={ignore_index}")
        for i in range(n):
            coord = tuple(idxs[i].tolist())
            print(f"  at {coord} -> value={int(vals[i])}")
        raise ValueError(f"{name}: found {idxs.size(0)} labels out of range [0,{V-1}] (or <0) "
                         f"with ignore_index={ignore_index}")

def mask_tokens(x: torch.Tensor,
                pad_id: int,
                mask_id: int,
                p: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BERT-style masking.
    x: (B,L) int
    returns: x_masked (B,L), labels (B,L) with -100 at unmasked, mask_bool (B,L)
    """
    B, L = x.shape
    labels = torch.full_like(x, -100)
    can_mask = x.ne(pad_id)
    sel = (torch.rand(B, L, device=x.device) < p) & can_mask
    labels[sel] = x[sel]

    x_masked = x.clone()
    rnd = torch.rand(B, L, device=x.device)

    # 80% -> [MASK]
    p_mask = (rnd < 0.8) & sel
    x_masked[p_mask] = mask_id

    # 10% -> random token
    p_rand = (rnd >= 0.8) & (rnd < 0.9) & sel
    x_masked[p_rand] = torch.randint(7, 12, (p_rand.sum(),), device=x.device) # randomly replace the original with either "A", "T", "C", "G", "N"

    # 10% -> keep original already handled by not p_mask & not p_rand
    return x_masked, labels

def mask_random_span(x: torch.Tensor,
                     attn_mask: torch.Tensor,
                     min_len: int,
                     max_len: int,
                     mask_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random contiguous span mask on valid tokens (attn_mask=1).
    Returns x_masked, labels (-100 elsewhere), start, end  (end exclusive).
    """
    B, L = x.shape
    x_masked = x.clone()
    labels = torch.full_like(x, -100)
    starts = torch.zeros(B, dtype=torch.long, device=x.device)
    ends   = torch.zeros(B, dtype=torch.long, device=x.device)

    valid_len = attn_mask.sum(dim=1)  # (B,)
    span_len = torch.randint(min_len, max_len+1, (B,), device=x.device)
    span_len = torch.minimum(span_len, valid_len)  # guard

    for b in range(B):
        Lb = int(valid_len[b].item())
        if Lb <= 0:
            continue
        max_start = max(1, Lb - int(span_len[b].item()))  # avoid negative
        s = torch.randint(0, max_start, (1,), device=x.device).item()
        e = s + int(span_len[b].item())
        starts[b] = s
        ends[b]   = e
        x_masked[b, s:e] = mask_id
        labels[b, s:e]   = x[b, s:e]
    return x_masked, labels, starts, ends

def mask_seed_span_mRNA(mrna_ids: torch.Tensor,
                         mrna_mask: torch.Tensor,
                         seed_start: torch.Tensor,
                         seed_end: torch.Tensor,
                         mask_id: int,):
    """
    Mask the provided seed span on mRNA.
    """
    B, Lk = mrna_ids.shape
    mrna_masked = mrna_ids.clone()
    labels = torch.full_like(mrna_ids, -100)

    for b in range(B):
        # Convert tensor indices to integers for indexing
        start_idx = int(seed_start[b].item())
        end_idx = int(seed_end[b].item())
        mrna_masked[b, start_idx:end_idx+1] = mask_id
        labels[b, start_idx:end_idx+1] = mrna_ids[b, start_idx:end_idx+1]
    return mrna_masked, labels

class TokenHead(torch.nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, embed_weight: torch.nn.Parameter):
        super().__init__()
        # transform block used by BERT
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.act = F.gelu
        self.layer_norm = nn.LayerNorm(embed_dim)
        # decoder: tie weight to embeddings
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        self.decoder.weight = embed_weight  # weight tying
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))  # separate bias

    def forward(self, hidden_states):                # (B, L, D)
        x = self.dense(hidden_states)                # (B, L, D)
        x = self.act(x)
        x = self.layer_norm(x)
        logits = self.decoder(x) + self.bias         # (B, L, V)
        return logits

class PairedPretrainWrapper(torch.nn.Module):
    """
    Assumes you already have:
      - self.embed_and_encode_mrna(ids, mask) -> H_mr: (B,Lq,D)
      - self.embed_and_encode_mirna(ids, mask)-> H_mi: (B,Lk,D)
      - self.cross_attend(Q, K, V, mask) -> (ctx, A) with ctx: (B,H,Lq,dh) or (B,Lq,D)

    We'll return logits for mRNA and miRNA.
    """
    def __init__(self, base_model, vocab_size, d_model, embed_weight):
        super().__init__()
        self.base = base_model
        self.mrna_head = TokenHead(d_model, vocab_size, embed_weight=embed_weight)
        self.mirna_head = TokenHead(d_model, vocab_size, embed_weight=embed_weight)

    def forward_pair(self, mrna_ids, mrna_mask, mirna_ids, mirna_mask, attn_mask=None):
        # 1) encode separately
        mirna_sn_embedding = self.base.predictor.sn_embedding(mirna_ids)
        mrna_sn_embedding = self.base.predictor.sn_embedding(mrna_ids)
        
        # add N-gram CNN-encoded embedding
        mirna_cnn_embedding = self.base.predictor.cnn_embedding(mirna_sn_embedding.transpose(-1, -2)) # (batch_size, embed_dim, mirna_len)
        mrna_cnn_embedding  = self.base.predictor.cnn_embedding(mrna_sn_embedding.transpose(-1, -2))  # (batch_size, embed_dim, mrna_len)
        mirna_embedding     = mirna_sn_embedding + mirna_cnn_embedding # (batch_size, mirna_len, embed_dim)
        mrna_embedding      = mrna_sn_embedding + mrna_cnn_embedding # (batch_size, mrna_len, embed_dim)

        mirna_embedding = self.base.predictor.mirna_encoder(mirna_embedding, mask=mirna_mask)  # (batch_size, mirna_len, embed_dim)
        mrna_embedding = self.base.predictor.mrna_encoder(mrna_embedding, mask=mrna_mask) # (batch_size, mrna_len, embed_dim)

        # 2) cross-attn (sliding window) to enrich mRNA from miRNA
        ctx_mr, attn_weights = self.base.predictor.cross_attn_layer(
                query=mrna_embedding,
                key=mirna_embedding,
                value=mirna_embedding,
                attention_mask=mirna_mask,
                query_attention_mask=mrna_mask,
            )
        self.cross_attn_output = ctx_mr
        ctx_mr_res = self.base.predictor.dropout(ctx_mr) + mrna_embedding # residual connection
        ctx_mr_norm = self.base.predictor.cross_norm(ctx_mr_res)
        ctx_mr_norm = ctx_mr_norm.masked_fill(mrna_mask.unsqueeze(-1)==0, 0) # (batch_size, mrna_len, embed_dim)

        # 3) cross-attn to enrich miRNA from mRNA
        ctx_mir, _ = self.base.predictor.cross_attn_layer(
            query=mirna_embedding,
            key=mrna_embedding,
            value=mrna_embedding,
            attention_mask=mrna_mask,
            query_attention_mask=mirna_mask,
        )
        ctx_mir_res = self.base.predictor.dropout(ctx_mir) + mirna_embedding # residual connection
        ctx_mir_norm = self.base.predictor.cross_norm(ctx_mir_res)
        ctx_mir_norm = ctx_mir_norm.masked_fill(mirna_mask.unsqueeze(-1)==0, 0) # (batch_size, mirna_len, embed_dim)

        logits_mr = self.mrna_head(ctx_mr_norm)  # (B,Lq,V)
        logits_mi = self.mirna_head(ctx_mir_norm) # (B,Lk,V)
        return logits_mr, logits_mi, attn_weights

def loss_stage1_baseline(wrapper: PairedPretrainWrapper, batch, pad_id, mask_id, vocab_size, evaluate=False):
    mi_in, mi_mask = batch["mirna_input_ids"], batch["mirna_attention_mask"]
    mr_in, mr_mask = batch["mrna_input_ids"],  batch["mrna_attention_mask"]

    mi_x, mi_y = mask_tokens(mi_in, pad_id, mask_id, p=0.15) # (B, L)
    mr_x, mr_y = mask_tokens(mr_in, pad_id, mask_id, p=0.15)

    logits_mr, logits_mi, _ = wrapper.forward_pair(mr_x, mr_mask, mi_x, mi_mask) # (B, L, V)

    loss_mr = F.cross_entropy(logits_mr.view(-1, vocab_size), mr_y.view(-1), ignore_index=-100) 
    loss_mi = F.cross_entropy(logits_mi.view(-1, vocab_size), mi_y.view(-1), ignore_index=-100)
    total_loss = loss_mr + loss_mi

    if evaluate:
        probs_mr, probs_mi = F.softmax(logits_mr, dim=-1), F.softmax(logits_mi, dim=-1) # (B, L, V)
        preds_mr, preds_mi = torch.argmax(probs_mr, dim=-1), torch.argmax(probs_mi, dim=-1) # (B, L)
        masked_mr, masked_mi = mr_y.ne(-100), mi_y.ne(-100) 
        correct_mr = 0
        if masked_mr.any():
            correct_mr = (preds_mr[masked_mr] == mr_y[masked_mr]).sum().item()
        correct_mi = 0
        if masked_mi.any():
            correct_mi = (preds_mi[masked_mi] == mi_y[masked_mi]).sum().item()
        total_correct = correct_mr + correct_mi
        return total_loss, total_correct, mr_y, mi_y
    else:
        return total_loss

def loss_stage2_seed_mrna(wrapper: PairedPretrainWrapper, batch, mask_id, vocab_size, evaluate=False):
    mi_in, mi_mask = batch["mirna_input_ids"], batch["mirna_attention_mask"]
    mr_in, mr_mask = batch["mrna_input_ids"],  batch["mrna_attention_mask"]
    seed_s, seed_e = batch["start_positions"], batch["end_positions"]  # (B,)

    mr_x, mr_y = mask_seed_span_mRNA(mr_in, mr_mask, seed_s, seed_e, mask_id)
    # (Optionally mask a small window on mRNA too, but spec asked to mask seeds from miRNA)
    logits_mr, _, _ = wrapper.forward_pair(mr_x, mr_mask, mi_in, mi_mask)

    # Only supervise the masked miRNA tokens
    loss_mr = F.cross_entropy(logits_mr.view(-1, vocab_size), mr_y.view(-1), ignore_index=-100)

    if evaluate:
        probs_mr = F.softmax(logits_mr, dim=-1)
        preds_mr = torch.argmax(probs_mr, dim=-1)
        masked = mr_y.ne(-100)
        if masked.any():
            correct_mr = (preds_mr[masked] == mr_y[masked]).sum().item()
            return loss_mr, correct_mr, mr_y
        else:
            return loss_mr, 0, mr_y
    else:
        return loss_mr

def loss_stage3_bispan(wrapper: PairedPretrainWrapper, batch, pad_id, mask_id, vocab_size, evaluate=False):
    mi_in, mi_mask = batch["mirna_input_ids"], batch["mirna_attention_mask"]
    mr_in, mr_mask = batch["mrna_input_ids"],  batch["mrna_attention_mask"]

    mi_x, mi_y, _, _ = mask_random_span(mi_in, mi_mask, 6, 8, mask_id)
    mr_x, mr_y, _, _ = mask_random_span(mr_in, mr_mask, 6, 20, mask_id)

    logits_mr, logits_mi, _ = wrapper.forward_pair(mr_x, mr_mask, mi_x, mi_mask)

    loss_mr = F.cross_entropy(logits_mr.view(-1, vocab_size), mr_y.view(-1), ignore_index=-100)
    loss_mi = F.cross_entropy(logits_mi.view(-1, vocab_size), mi_y.view(-1), ignore_index=-100)
    total_loss = loss_mr + loss_mi

    if evaluate:
        probs_mr, probs_mi = F.softmax(logits_mr, dim=-1), F.softmax(logits_mi, dim=-1)
        preds_mr, preds_mi = torch.argmax(probs_mr, dim=-1), torch.argmax(probs_mi, dim=-1)
        
        masked_mr = mr_y.ne(-100)
        correct_mr = 0
        if masked_mr.any():
            correct_mr = (preds_mr[masked_mr] == mr_y[masked_mr]).sum().item()

        masked_mi = mi_y.ne(-100)
        correct_mi = 0
        if masked_mi.any():
            correct_mi = (preds_mi[masked_mi] == mi_y[masked_mi]).sum().item()
            
        total_correct = correct_mr + correct_mi
        return total_loss, total_correct, mr_y, mi_y
    else:
        return total_loss

def run_pretrain_epoch(stage, wrapper, dataloader, optimizer, device,
                       pad_id, mask_id, vocab_size):
    wrapper.train()
    total = 0.0
    for batch in dataloader:
        for k in batch: batch[k] = batch[k].to(device)
        loss1 = loss_stage1_baseline(wrapper, batch, pad_id, mask_id)
        loss2 = loss_stage2_seed_mrna(wrapper, batch, mask_id, vocab_size)
        loss3 = loss_stage3_bispan(wrapper, batch, pad_id, mask_id, vocab_size)

        loss = loss1 + loss2 + loss3
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(dataloader))