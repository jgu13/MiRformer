#!/usr/bin/env python3
"""
Simple test to check if the tokenizer can be imported and initialized.
"""

import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    print("Attempting to import CharacterTokenizer...")
    from Data_pipeline import CharacterTokenizer
    print("✅ Successfully imported CharacterTokenizer")
    
    print("Attempting to initialize tokenizer...")
    tokenizer = CharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],
        model_max_length=100
    )
    print("✅ Successfully initialized tokenizer")
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("🎉 Basic initialization test passed!")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
