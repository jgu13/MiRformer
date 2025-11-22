#!/bin/bash

# Example script to run DDP training
# This script demonstrates how to use the modified Pretrain.py with DDP

echo "Starting DDP training example..."

# To run training, simply execute the script
# The parameters are set directly in the main() function in Pretrain.py
echo "Running training with parameters set in Pretrain.py..."
python scripts/Pretrain.py

echo "Training completed!"
echo ""
echo "To modify training parameters, edit the main() function in Pretrain.py:"
echo "- Set use_ddp = True for DDP training, False for single GPU"
echo "- Adjust epochs, batch_size, embed_dim, etc. as needed"
