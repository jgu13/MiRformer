"""
Test the Attention regularization implementation.
"""
import torch
from Attention_regularization import _row_normalize, kl_diag_seed_loss

B = 1
H = 1
Lq = 10
Lk = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_mask = torch.ones((B,Lq), device=device)
k_mask = torch.ones((B,Lk), device=device)
seed_q_start = torch.full((B,), 2, device=device)
seed_q_end = torch.full((B,), 8, device=device)
y_pos = torch.ones((B,), device=device)

# 1) Pass a “perfect diagonal” A and see KL ≈ 0
A = torch.zeros(B,H,Lq,Lk, device=device)
for b in range(B):
    s, e = int(seed_q_start[b]), int(seed_q_end[b])
    j = 1  # k_seed_start
    for i in range(s, e+1):
        A[b,:,i,j+(i-s)] = 1.0
print(A)
A = _row_normalize(A, k_mask[:,None,None,:].float())
print(kl_diag_seed_loss(A, seed_q_start, seed_q_end, q_mask, k_mask, y_pos, k_seed_start=1))  # should be near 0