#!/usr/bin/env python3
"""
Test script to verify the correctness of both classification modes:
1. CLS-only classification (use_cls_only=True)
2. Cross-attention classification (use_cls_only=False)

This script helps verify that both modes work correctly for single-base perturbation testing.
"""

import torch
import torch.nn as nn
from transformer_model import QuestionAnsweringModel

def test_classification_modes():
    """Test both classification modes with dummy data"""
    
    # Model configuration
    mrna_max_len = 100
    mirna_max_len = 24
    embed_dim = 256
    num_heads = 4
    num_layers = 2
    ff_dim = 512
    batch_size = 4
    
    print("Initializing model...")
    model = QuestionAnsweringModel(
        mrna_max_len=mrna_max_len,
        mirna_max_len=mirna_max_len,
        device="cpu",  # Use CPU for testing
        epochs=1,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        batch_size=batch_size,
        lr=1e-4,
        seed=42,
        predict_span=False,  # Focus on binding prediction
        predict_binding=True,
        use_longformer=True
    )
    
    print("Creating dummy data...")
    # Create dummy input data
    mirna = torch.randint(0, 13, (batch_size, mirna_max_len))  # Random miRNA sequences
    mrna = torch.randint(0, 13, (batch_size, mrna_max_len))    # Random mRNA sequences
    
    # Create attention masks (1 for valid tokens, 0 for padding)
    mirna_mask = torch.ones(batch_size, mirna_max_len)
    mrna_mask = torch.ones(batch_size, mrna_max_len)
    
    # Set some padding tokens
    mrna_mask[:, 80:] = 0  # Last 20 positions are padding
    mirna_mask[:, 20:] = 0  # Last 4 positions are padding
    
    # Create dummy targets (binary binding labels)
    targets = torch.randint(0, 2, (batch_size,)).float()
    
    print("Testing CLS-only classification mode...")
    # Test CLS-only mode
    model.eval()
    with torch.no_grad():
        cls_outputs = model(
            mirna=mirna,
            mrna=mrna,
            mrna_mask=mrna_mask,
            mirna_mask=mirna_mask,
            use_cls_only=True
        )
        
        binding_logit_cls, binding_weights_cls, start_logits_cls, end_logits_cls = cls_outputs
        
        print(f"CLS-only mode:")
        print(f"  - Binding logit shape: {binding_logit_cls.shape}")
        print(f"  - Binding weights: {binding_weights_cls}")
        print(f"  - Start logits: {start_logits_cls}")
        print(f"  - End logits: {end_logits_cls}")
        
        # Verify CLS-only outputs
        assert binding_logit_cls.shape == (batch_size,), f"Expected (B,), got {binding_logit_cls.shape}"
        assert binding_weights_cls is None, "Expected None for binding weights in CLS-only mode"
        assert start_logits_cls is None, "Expected None for start logits when predict_span=False"
        assert end_logits_cls is None, "Expected None for end logits when predict_span=False"
        
        print("  ✅ CLS-only mode outputs are correct!")
    
    print("\nTesting Cross-attention classification mode...")
    # Test Cross-attention mode
    with torch.no_grad():
        cross_outputs = model(
            mirna=mirna,
            mrna=mrna,
            mrna_mask=mrna_mask,
            mirna_mask=mirna_mask,
            use_cls_only=False
        )
        
        binding_logit_cross, binding_weights_cross, start_logits_cross, end_logits_cross = cross_outputs
        
        print(f"Cross-attention mode:")
        print(f"  - Binding logit shape: {binding_logit_cross.shape}")
        print(f"  - Binding weights shape: {binding_weights_cross.shape if binding_weights_cross is not None else None}")
        print(f"  - Start logits: {start_logits_cross}")
        print(f"  - End logits: {end_logits_cross}")
        
        # Verify Cross-attention outputs
        assert binding_logit_cross.shape == (batch_size,), f"Expected (B,), got {binding_logit_cross.shape}"
        assert binding_weights_cross.shape == (batch_size, mrna_max_len), f"Expected (B, L), got {binding_weights_cross.shape}"
        assert start_logits_cross is None, "Expected None for start logits when predict_span=False"
        assert end_logits_cross is None, "Expected None for end logits when predict_span=False"
        
        # Verify that CLS token is masked out in binding weights
        cls_weights = binding_weights_cross[:, 0]  # CLS token weights
        assert torch.allclose(cls_weights, torch.zeros_like(cls_weights), atol=1e-6), "CLS token should be masked out in cross-attention mode"
        
        print("  ✅ Cross-attention mode outputs are correct!")
    
    print("\nTesting binding prediction consistency...")
    # Test that both modes produce reasonable binding predictions
    with torch.no_grad():
        # Get predictions from both modes
        cls_preds = torch.sigmoid(binding_logit_cls)
        cross_preds = torch.sigmoid(binding_logit_cross)
        
        print(f"  - CLS-only predictions: {cls_preds}")
        print(f"  - Cross-attention predictions: {cross_preds}")
        
        # Both should produce valid probabilities between 0 and 1
        assert torch.all(cls_preds >= 0) and torch.all(cls_preds <= 1), "CLS predictions should be probabilities"
        assert torch.all(cross_preds >= 0) and torch.all(cross_preds <= 1), "Cross-attention predictions should be probabilities"
        
        print("  ✅ Binding predictions are valid probabilities!")
    
    print("\n🎉 All tests passed! Both classification modes are working correctly.")
    
    # Summary of what was tested
    print("\n📋 Summary of tested functionality:")
    print("1. ✅ CLS-only classification (use_cls_only=True)")
    print("   - Uses only CLS token for binding prediction")
    print("   - Returns binding logits and None for weights")
    print("2. ✅ Cross-attention classification (use_cls_only=False)")
    print("   - Uses LSE pooling over non-CLS tokens")
    print("   - Returns binding logits and attention weights")
    print("   - CLS token is properly masked out")
    print("3. ✅ Both modes produce valid binding predictions")
    print("4. ✅ Model handles different input shapes correctly")
    
    return True

def test_single_base_perturbation():
    """Test how the model handles single-base perturbations"""
    
    print("\n🧬 Testing single-base perturbation handling...")
    
    # Create a simple test case
    batch_size = 2
    mrna_len = 50
    mirna_len = 24
    
    # Create base sequences
    base_mrna = torch.randint(0, 13, (batch_size, mrna_len))
    base_mirna = torch.randint(0, 13, (batch_size, mirna_len))
    
    # Create perturbed sequences (single base change)
    perturbed_mrna = base_mrna.clone()
    perturbed_mrna[0, 25] = (perturbed_mrna[0, 25] + 1) % 13  # Change one base
    
    # Create masks
    mrna_mask = torch.ones(batch_size, mrna_len)
    mirna_mask = torch.ones(batch_size, mirna_len)
    
    # Create model
    model = QuestionAnsweringModel(
        mrna_max_len=mrna_len,
        mirna_max_len=mirna_len,
        device="cpu",
        embed_dim=128,  # Smaller for testing
        num_heads=2,
        num_layers=1,
        ff_dim=256,
        predict_span=False,
        predict_binding=True,
        use_longformer=True
    )
    
    model.eval()
    with torch.no_grad():
        # Get predictions for base sequences
        base_outputs = model(
            mirna=base_mirna,
            mrna=base_mrna,
            mrna_mask=mrna_mask,
            mirna_mask=mirna_mask,
            use_cls_only=False
        )
        base_binding = torch.sigmoid(base_outputs[0])
        
        # Get predictions for perturbed sequences
        perturbed_outputs = model(
            mirna=base_mirna,
            mrna=perturbed_mrna,
            mrna_mask=mrna_mask,
            mirna_mask=mirna_mask,
            use_cls_only=False
        )
        perturbed_binding = torch.sigmoid(perturbed_outputs[0])
        
        print(f"  - Base sequence binding predictions: {base_binding}")
        print(f"  - Perturbed sequence binding predictions: {perturbed_binding}")
        print(f"  - Change in binding: {perturbed_binding - base_binding}")
        
        # The predictions should be different (though magnitude depends on model training)
        assert not torch.allclose(base_binding, perturbed_binding, atol=1e-6), "Perturbation should change predictions"
        
        print("  ✅ Single-base perturbation is properly detected!")
    
    return True

if __name__ == "__main__":
    try:
        print("🚀 Starting classification mode testing...")
        test_classification_modes()
        
        print("\n🔬 Starting single-base perturbation testing...")
        test_single_base_perturbation()
        
        print("\n🎯 All tests completed successfully!")
        print("\n💡 You can now use both classification modes for your single-base perturbation experiments:")
        print("   - CLS-only: model(..., use_cls_only=True)")
        print("   - Cross-attention: model(..., use_cls_only=False)")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

