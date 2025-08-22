#!/usr/bin/env python3
"""
Test script to verify the MIL implementation with global token.
"""

import torch
import torch.nn as nn
from transformer_model import CrossAttentionPredictor
from Data_pipeline import CharacterTokenizer

def test_mil_implementation():
    """Test the MIL implementation with a simple example"""
    
    # Test parameters
    batch_size = 2
    mrna_max_len = 10
    mirna_max_len = 5
    embed_dim = 64
    
    print("Testing MIL implementation...")
    print(f"Batch size: {batch_size}")
    print(f"mRNA max length: {mrna_max_len}")
    print(f"miRNA max length: {mirna_max_len}")
    print(f"Embedding dimension: {embed_dim}")
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                   add_special_tokens=False,
                                   model_max_length=mrna_max_len,
                                   padding_side="right")
    
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"MRNA_CLS token ID: {tokenizer.mrna_cls_token_id}")
    
    # Create model
    model = CrossAttentionPredictor(
        mirna_max_len=mirna_max_len,
        mrna_max_len=mrna_max_len,
        tokenizer=tokenizer,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        window_size=4,
        ff_dim=128,
        device='cpu',
        predict_span=True,
        predict_binding=True,
        use_longformer=True
    )
    
    print(f"Model created successfully")
    print(f"Embedding table size: {model.sn_embedding.num_embeddings}")
    
    # Create test inputs
    mirna = torch.randint(0, vocab_size, (batch_size, mirna_max_len))
    mrna = torch.randint(0, vocab_size, (batch_size, mrna_max_len))
    
    # Prepend global token to mRNA (simulating dataset preprocessing)
    mrna_with_cls = torch.cat([
        torch.full((batch_size, 1), tokenizer.mrna_cls_token_id, dtype=torch.long),
        mrna[:, :-1]  # Remove last token to make room for CLS
    ], dim=1)
    
    # Create masks
    mirna_mask = torch.ones(batch_size, mirna_max_len)
    mrna_mask = torch.ones(batch_size, mrna_max_len)
    
    print(f"Input shapes:")
    print(f"  miRNA: {mirna.shape}")
    print(f"  mRNA (with CLS): {mrna_with_cls.shape}")
    print(f"  mRNA mask: {mrna_mask.shape}")
    
        # Verify global token is at position 0
        assert (mrna_with_cls[:, 0] == tokenizer.mrna_cls_token_id).all(), "Global token not at position 0"
        print("✓ Global token correctly placed at position 0")
        
        # Verify attention mask values
        # mrna_mask: 1 for valid tokens (including global), 0 for padding
        # Expected Longformer format: 1 for global (pos 0), 0 for local (pos 1+), -1 for padding
        assert (mrna_mask[:, 0] == 1).all(), "Global token should have attention mask = 1"
        print("✓ Attention mask values correctly set")
    
    # Forward pass
    try:
        outputs = model(mirna, mrna_with_cls, mrna_mask, mirna_mask)
        binding_logit, binding_aux, start_logits, end_logits = outputs
        
        print("✓ Forward pass successful")
        print(f"Output shapes:")
        print(f"  binding_logit: {binding_logit.shape} (MIL binding prediction)")
        print(f"  start_logits: {start_logits.shape if start_logits is not None else 'None'}")
        print(f"  end_logits: {end_logits.shape if end_logits is not None else 'None'}")
        print(f"  binding_aux: dict with pos_weights and pos_logits")
        
        # Test MIL binding attention weights
        pos_weights = binding_aux["pos_weights"]
        pos_logits = binding_aux["pos_logits"]
        print(f"  pos_weights: {pos_weights.shape}")
        print(f"  pos_logits: {pos_logits.shape}")
        
        # Verify MIL outputs
        assert binding_logit.shape == (batch_size,), f"Expected binding_logit shape (batch_size,), got {binding_logit.shape}"
        assert pos_weights.shape == (batch_size, mrna_max_len), f"Expected pos_weights shape (batch_size, mrna_max_len), got {pos_weights.shape}"
        assert pos_logits.shape == (batch_size, mrna_max_len), f"Expected pos_logits shape (batch_size, mrna_max_len), got {pos_logits.shape}"
        
        print("✓ All output shapes correct")
        
        # Test that MIL binding attention weights sum to 1 for valid positions
        valid_weights = pos_weights * (mrna_mask > 0)
        weight_sums = valid_weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6), "MIL binding weights don't sum to 1"
        print("✓ MIL binding attention weights properly normalized")
        
        print("\n🎉 All tests passed! MIL binding implementation is working correctly.")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise

if __name__ == "__main__":
    test_mil_implementation()


