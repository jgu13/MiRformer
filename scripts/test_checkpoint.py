#!/usr/bin/env python3
"""
Test script to verify checkpoint functionality works correctly.
This script tests the checkpoint saving and loading without running the full training.
"""

import os
import sys
import torch
import tempfile
import shutil
from ckpt_utils import save_training_state, load_training_state, latest_checkpoint

def test_checkpoint_functionality():
    """Test checkpoint saving and loading functionality."""
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="checkpoint_test_")
    print(f"Testing checkpoint functionality in: {test_dir}")
    
    try:
        # Create a simple model for testing
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        # Test saving checkpoint
        best_metrics = {"overall_acc": 0.85, "loss": 0.15}
        save_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            step=1000,
            best_metrics=best_metrics,
            ckpt_dir=test_dir,
            tag="test"
        )
        
        print("✓ Checkpoint saved successfully")
        
        # Test loading checkpoint
        loaded_state = load_training_state(
            ckpt_path=os.path.join(test_dir, "test.pth"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu"
        )
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  - Epoch: {loaded_state['epoch']}")
        print(f"  - Step: {loaded_state['step']}")
        print(f"  - Best metrics: {loaded_state['best_metrics']}")
        
        # Test latest checkpoint detection
        latest_ckpt = latest_checkpoint(test_dir)
        print(f"✓ Latest checkpoint found: {latest_ckpt}")
        
        print("\n🎉 All checkpoint tests passed!")
        
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory: {test_dir}")
    
    return True

if __name__ == "__main__":
    success = test_checkpoint_functionality()
    sys.exit(0 if success else 1)
