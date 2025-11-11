import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import load_dataset
from Data_pipeline import QuestionAnswerDataset, CharacterTokenizer
from transformer_model import QuestionAnsweringModel

PROJ_HOME = os.path.expanduser("~/projects/mirLM")


def create_ground_truth_labels(start_positions, end_positions, mrna_mask):
    """
    Create ground truth labels: 1.0 from seed start to seed end, 0.0 elsewhere.
    
    Args:
        start_positions: (B,) tensor of start positions (0-indexed token positions)
        end_positions: (B,) tensor of end positions (0-indexed token positions)
        mrna_mask: (B, L) tensor where 1 indicates valid tokens, 0 indicates padding
    
    Returns:
        labels: (B, L) tensor with 1.0 in [start, end] range, 0.0 elsewhere
    """
    batch_size, max_len = mrna_mask.shape
    labels = torch.zeros(batch_size, max_len, dtype=torch.float32, device=mrna_mask.device)
    
    for i in range(batch_size):
        start = start_positions[i].item()
        end = end_positions[i].item()
        seq_len = mrna_mask[i].sum().item()  # Actual sequence length (excluding padding)
        
        # Only create labels for valid positions (within actual sequence length)
        if start >= 0 and end >= 0 and start < seq_len and end < seq_len and start <= end:
            # Set 1.0 from start to end (inclusive)
            labels[i, start:end+1] = 1.0
    
    return labels


def evaluate_span_auroc(model, dataloader, device):
    """
    Evaluate span AUROC score.
    
    Steps:
    1. Get start_logits and end_logits from model
    2. Apply sigmoid to both
    3. Take max of the two
    4. Compare with ground truth labels (1.0 from seed start to seed end, 0.0 elsewhere)
    5. Calculate AUROC
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            for k in batch:
                batch[k] = batch[k].to(device)
            
            mrna_mask = batch["mrna_attention_mask"]
            mirna_mask = batch["mirna_attention_mask"]
            
            # Get model predictions
            outputs = model(
                mirna=batch["mirna_input_ids"],
                mrna=batch["mrna_input_ids"],
                mrna_mask=mrna_mask,
                mirna_mask=mirna_mask,
            )
            
            binding_logit, binding_weights, start_logits, end_logits, cleavage_logits = outputs
            
            # Mask padded positions
            start_logits = start_logits.masked_fill(mrna_mask == 0, float("-inf"))
            end_logits = end_logits.masked_fill(mrna_mask == 0, float("-inf"))
            
            # Step 2: Apply sigmoid to both start and end logits
            start_probs = torch.sigmoid(start_logits)  # (B, L)
            end_probs = torch.sigmoid(end_logits)      # (B, L)
            
            # Step 3: take the mean of start and end probs
            max_probs = torch.mean(torch.stack([start_probs, end_probs]), dim=0)  # (B, L)
            
            # Step 4: Create ground truth labels
            start_positions = batch["start_positions"]  # (B,)
            end_positions = batch["end_positions"]      # (B,)
            
            ground_truth = create_ground_truth_labels(
                start_positions, end_positions, mrna_mask
            )  # (B, L)
            
            # Flatten predictions and labels for AUROC calculation
            # Only consider valid positions (non-padded)
            batch_size = batch["mrna_input_ids"].shape[0]
            for i in range(batch_size):
                seq_len = mrna_mask[i].sum().item()  # Actual sequence length
                # Extract predictions and labels for valid positions only
                pred_seq = max_probs[i, :seq_len].cpu().numpy()
                label_seq = ground_truth[i, :seq_len].cpu().numpy()
                
                all_predictions.extend(pred_seq)
                all_labels.extend(label_seq)
    
    # Step 5: Calculate AUROC
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Filter out cases where all labels are 0 (no positive examples in sequence)
    # This can happen if start/end positions are invalid
    if len(all_labels) == 0 or all_labels.sum() == 0:
        print("Warning: No positive labels found. Cannot calculate AUROC.")
        return None
    
    try:
        auroc = roc_auc_score(all_labels, all_predictions)
        return auroc
    except ValueError as e:
        print(f"Error calculating AUROC: {e}")
        print(f"Unique labels: {np.unique(all_labels)}")
        print(f"Predictions range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
        return None


def main():
    # Configuration
    ckpt_path = "/home/mcb/users/jgu13/projects/mirLM/checkpoints/TargetScan/TwoTowerTransformer/Longformer/520/embed=1024d/norm_by_query/LSE/best_composite_0.9312_0.9975_epoch19.pth"
    data_path = os.path.join(PROJ_HOME, "TargetScan_dataset/TargetScan_validation_500_randomized_start.csv")
    
    mrna_max_len = 520
    mirna_max_len = 24
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    D_val = load_dataset(data_path, sep=',')
    # filter for positive samples
    D_val = D_val[D_val["label"] == 1]
    
    tokenizer = CharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],
        model_max_length=mrna_max_len,
        padding_side="right"
    )
    
    ds_val = QuestionAnswerDataset(
        data=D_val,
        tokenizer=tokenizer,
        mrna_max_len=mrna_max_len,
        mirna_max_len=mirna_max_len,
        seed_start_col="seed start" if "seed start" in D_val.columns else None,
        seed_end_col="seed end" if "seed end" in D_val.columns else None,
    )
    
    val_loader = DataLoader(
        ds_val,
        batch_size=32,
        shuffle=False,
    )
    
    # Initialize model
    print("Initializing model...")
    model = QuestionAnsweringModel(
        mrna_max_len=mrna_max_len,
        mirna_max_len=mirna_max_len,
        device=device,
        epochs=100,
        embed_dim=1024,
        num_heads=8,
        num_layers=4,
        ff_dim=4096,
        batch_size=32,
        lr=3e-5,
        seed=42,
        predict_span=True,
        predict_binding=False,
        predict_cleavage=False,
        use_longformer=True
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}...")
    loaded_state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(loaded_state_dict, strict=False)
    model.to(device)
    print("Checkpoint loaded successfully.")
    
    # Evaluate
    print("Evaluating span AUROC...")
    auroc = evaluate_span_auroc(model, val_loader, device)
    
    if auroc is not None:
        print(f"\nSpan AUROC Score: {auroc:.4f}")
        # save the auroc score to a file
        with open("span_auroc.txt", "w") as f:
            f.write(f"Span AUROC Score: {auroc:.4f}")
    else:
        print("\nFailed to calculate AUROC.")


if __name__ == "__main__":
    main()

