#!/usr/bin/env python3
"""
Final test to verify that the simplified tokenizer works with the transformer model.
"""

import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    print("Testing simplified tokenizer integration...")
    
    # Test 1: Import the simplified tokenizer
    from Data_pipeline_simple import SimpleCharacterTokenizer
    print("✅ Successfully imported SimpleCharacterTokenizer")
    
    # Test 2: Initialize tokenizer
    tokenizer = SimpleCharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],
        model_max_length=100
    )
    print("✅ Successfully initialized SimpleCharacterTokenizer")
    
    # Test 3: Test basic functionality
    test_text = "ATCGN"
    encoded = tokenizer.encode_plus(test_text)
    decoded = tokenizer.decode(encoded["input_ids"])
    print(f"  - Test text: '{test_text}'")
    print(f"  - Encoded: {encoded}")
    print(f"  - Decoded: '{decoded}'")
    assert decoded == test_text, "Encoding/decoding mismatch"
    
    # Test 4: Test vocabulary
    vocab = tokenizer.get_vocab()
    print(f"  - Vocabulary size: {tokenizer.vocab_size}")
    print(f"  - CLS token ID: {tokenizer.cls_token_id}")
    print(f"  - MRNA_CLS token ID: {tokenizer.mrna_cls_token_id}")
    
    print("\n🎉 Simplified tokenizer is working correctly!")
    print("\nNow testing integration with transformer model...")
    
    # Test 5: Test that the transformer model can import the tokenizer
    try:
        # This should now work without the NotImplementedError
        from transformer_model import QuestionAnsweringModel
        print("✅ Successfully imported QuestionAnsweringModel")
        
        # Test 6: Test model initialization with simplified tokenizer
        model = QuestionAnsweringModel(
            mrna_max_len=100,
            mirna_max_len=24,
            device="cpu",  # Use CPU for testing
            epochs=1,
            embed_dim=256,
            num_heads=4,
            num_layers=2,
            ff_dim=512,
            batch_size=4,
            lr=1e-4,
            seed=42,
            predict_span=False,
            predict_binding=True,
            use_longformer=True
        )
        print("✅ Successfully initialized QuestionAnsweringModel")
        
        print("\n🎯 All tests passed! The simplified tokenizer integration is working.")
        print("\nYou can now run your CLS-only training without the NotImplementedError.")
        
    except Exception as e:
        print(f"❌ Transformer model test failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
