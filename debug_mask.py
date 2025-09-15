#!/usr/bin/env python3

import torch

# Simulate the mask combination logic
def test_mask_combination():
    # Simulate some mask values
    bsz, q_len, k_len = 2, 10, 8
    
    # Case 1: Both masks have valid tokens
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],  # mirna_mask
                                  [1, 1, 1, 1, 1, 0, 0, 0]])
    query_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # mrna_mask
                                        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    
    print("=== Case 1: Both masks have valid tokens ===")
    print(f"attention_mask shape: {attention_mask.shape}, values: {attention_mask}")
    print(f"query_attention_mask shape: {query_attention_mask.shape}, values: {query_attention_mask}")
    
    attention_mask_bool = (attention_mask > 0)
    query_attention_mask_bool = (query_attention_mask > 0)
    
    print(f"attention_mask_bool: {attention_mask_bool}")
    print(f"query_attention_mask_bool: {query_attention_mask_bool}")
    
    # Create combined mask
    mask = attention_mask_bool[:, None, :] & query_attention_mask_bool[:, :, None]
    print(f"Combined mask shape: {mask.shape}")
    print(f"Combined mask any True: {mask.any()}")
    print(f"Combined mask all True: {mask.all()}")
    
    # Case 2: One mask has all zeros
    print("\n=== Case 2: One mask has all zeros ===")
    attention_mask_zero = torch.zeros_like(attention_mask)
    query_attention_mask_zero = torch.zeros_like(query_attention_mask)
    
    print(f"attention_mask_zero: {attention_mask_zero}")
    print(f"query_attention_mask_zero: {query_attention_mask_zero}")
    
    attention_mask_zero_bool = (attention_mask_zero > 0)
    query_attention_mask_zero_bool = (query_attention_mask_zero > 0)
    
    print(f"attention_mask_zero_bool: {attention_mask_zero_bool}")
    print(f"query_attention_mask_zero_bool: {query_attention_mask_zero_bool}")
    
    mask_zero = attention_mask_zero_bool[:, None, :] & query_attention_mask_zero_bool[:, :, None]
    print(f"Combined mask_zero any True: {mask_zero.any()}")
    print(f"Combined mask_zero all True: {mask_zero.all()}")

if __name__ == "__main__":
    test_mask_combination()
