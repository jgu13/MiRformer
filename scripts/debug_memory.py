#!/usr/bin/env python3
"""
Debug script to check memory usage and identify potential OOM issues.
"""

import os
import psutil
import torch
import gc

def check_system_memory():
    """Check system memory usage."""
    memory = psutil.virtual_memory()
    print(f"System Memory:")
    print(f"  Total: {memory.total / (1024**3):.2f} GB")
    print(f"  Available: {memory.available / (1024**3):.2f} GB")
    print(f"  Used: {memory.used / (1024**3):.2f} GB")
    print(f"  Percentage: {memory.percent:.1f}%")
    return memory.available / (1024**3)  # Return available GB

def check_gpu_memory():
    """Check GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 0
    
    print(f"GPU Memory:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        cached = torch.cuda.memory_reserved(i) / (1024**3)
        free = total_memory - cached
        
        print(f"  GPU {i} ({props.name}):")
        print(f"    Total: {total_memory:.2f} GB")
        print(f"    Allocated: {allocated:.2f} GB")
        print(f"    Cached: {cached:.2f} GB")
        print(f"    Free: {free:.2f} GB")
    
    return free

def estimate_model_memory(embed_dim=1024, num_layers=4, num_heads=8, ff_dim=4096, 
                         mrna_max_len=520, mirna_max_len=24, batch_size=32):
    """Estimate memory usage for the model."""
    print(f"Model Memory Estimation:")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  FF dim: {ff_dim}")
    print(f"  mRNA max len: {mrna_max_len}")
    print(f"  miRNA max len: {mirna_max_len}")
    print(f"  Batch size: {batch_size}")
    
    # Rough estimation (this is very approximate)
    # Each parameter is 4 bytes (float32)
    # We need to account for forward pass, backward pass, and optimizer states
    
    # Model parameters
    vocab_size = 5  # A, T, C, G, N
    embedding_params = vocab_size * embed_dim
    attention_params = num_layers * (4 * embed_dim * embed_dim + 2 * embed_dim)  # Q, K, V, O projections + bias
    ff_params = num_layers * (2 * embed_dim * ff_dim + ff_dim + embed_dim)  # FF layers
    total_params = embedding_params + attention_params + ff_params
    
    # Memory per sample (forward + backward + optimizer states)
    sequence_length = mrna_max_len + mirna_max_len
    memory_per_sample = sequence_length * embed_dim * 4 * 3  # 3x for forward, backward, optimizer
    
    total_memory_gb = (total_params * 4 + batch_size * memory_per_sample) / (1024**3)
    
    print(f"  Estimated model parameters: {total_params:,}")
    print(f"  Estimated memory per batch: {batch_size * memory_per_sample / (1024**3):.2f} GB")
    print(f"  Total estimated memory: {total_memory_gb:.2f} GB")
    
    return total_memory_gb

def suggest_memory_optimizations():
    """Suggest memory optimization strategies."""
    print(f"\nMemory Optimization Suggestions:")
    print(f"1. Reduce batch size (currently 32)")
    print(f"2. Reduce accumulation steps (currently 8)")
    print(f"3. Use gradient checkpointing")
    print(f"4. Reduce model dimensions (embed_dim, ff_dim)")
    print(f"5. Use mixed precision training")
    print(f"6. Reduce sequence lengths if possible")

def main():
    print("=== Memory Debug Information ===\n")
    
    # Check system memory
    available_ram = check_system_memory()
    print()
    
    # Check GPU memory
    available_gpu = check_gpu_memory()
    print()
    
    # Estimate model memory
    estimated_memory = estimate_model_memory()
    print()
    
    # Check if we have enough memory
    if available_ram < estimated_memory * 2:  # Need 2x for safety
        print(f"⚠️  WARNING: Insufficient RAM!")
        print(f"   Available: {available_ram:.2f} GB")
        print(f"   Estimated need: {estimated_memory * 2:.2f} GB")
    else:
        print(f"✓ RAM looks sufficient")
    
    if available_gpu < estimated_memory:
        print(f"⚠️  WARNING: Insufficient GPU memory!")
        print(f"   Available: {available_gpu:.2f} GB")
        print(f"   Estimated need: {estimated_memory:.2f} GB")
    else:
        print(f"✓ GPU memory looks sufficient")
    
    suggest_memory_optimizations()

if __name__ == "__main__":
    main()
