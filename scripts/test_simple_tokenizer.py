#!/usr/bin/env python3
"""
Test script for the simplified CharacterTokenizer.
"""

import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from Data_pipeline_simple import SimpleCharacterTokenizer
    print("✅ Successfully imported SimpleCharacterTokenizer")
    
    # Test tokenizer initialization
    print("Testing tokenizer initialization...")
    tokenizer = SimpleCharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],
        model_max_length=100
    )
    print("✅ Tokenizer initialized successfully")
    
    # Test basic functionality
    print("Testing basic tokenizer functionality...")
    
    # Test vocab size
    print(f"  - Vocabulary size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == 13, f"Expected vocab size 13, got {tokenizer.vocab_size}"
    
    # Test tokenization
    test_text = "ATCGN"
    tokens = tokenizer._tokenize(test_text)
    print(f"  - Tokenization of '{test_text}': {tokens}")
    assert tokens == ['A', 'T', 'C', 'G', 'N'], f"Expected ['A', 'T', 'C', 'G', 'N'], got {tokens}"
    
    # Test token to ID conversion
    token_ids = [tokenizer._convert_token_to_id(token) for token in tokens]
    print(f"  - Token IDs: {token_ids}")
    
    # Test ID to token conversion
    decoded_tokens = [tokenizer._convert_id_to_token(token_id) for token_id in token_ids]
    print(f"  - Decoded tokens: {decoded_tokens}")
    assert decoded_tokens == tokens, f"Expected {tokens}, got {decoded_tokens}"
    
    # Test get_vocab method
    vocab = tokenizer.get_vocab()
    print(f"  - Vocabulary: {vocab}")
    assert isinstance(vocab, dict), "get_vocab should return a dictionary"
    assert len(vocab) == tokenizer.vocab_size, "Vocabulary size mismatch"
    
    # Test encode_plus method
    encoded = tokenizer.encode_plus(test_text)
    print(f"  - Encoded output: {encoded}")
    assert "input_ids" in encoded, "encode_plus should return input_ids"
    assert "attention_mask" in encoded, "encode_plus should return attention_mask"
    
    # Test decode method
    decoded_text = tokenizer.decode(encoded["input_ids"])
    print(f"  - Decoded text: '{decoded_text}'")
    assert decoded_text == test_text, f"Expected '{test_text}', got '{decoded_text}'"
    
    # Test special token IDs
    print(f"  - CLS token ID: {tokenizer.cls_token_id}")
    print(f"  - MRNA_CLS token ID: {tokenizer.mrna_cls_token_id}")
    assert tokenizer.cls_token_id == 0, "CLS token ID should be 0"
    assert tokenizer.mrna_cls_token_id == 7, "MRNA_CLS token ID should be 7"
    
    print("🎉 All simplified tokenizer tests passed!")
    print("\nThe SimpleCharacterTokenizer is working correctly and can be used as a replacement.")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
