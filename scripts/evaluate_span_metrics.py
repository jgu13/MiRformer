
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from utils import load_dataset
from Data_pipeline import TokenClassificationDataset, CharacterTokenizer
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


def evaluate_span_metrics(model, dataloader, device):
    """
    Evaluate span AUROC, AUPRC, Precision, Recall, F1 score.
    
    Steps:
    1. Get start_logits and end_logits from model
    2. Apply sigmoid to both
    3. Take max of the two
    4. Compare with ground truth labels (1.0 from seed start to seed end, 0.0 elsewhere)
    5. Calculate AUROC, AUPRC, Precision, Recall, F1 score
    """
    model.eval()
    total_loss = 0.0
    all_token_probs = []  # Store probabilities for AUROC/AUPRC
    all_token_preds = []  # Store argmax predictions (after B->I conversion)
    all_token_labels = []
    all_binding_preds = []
    all_binding_labels = []

    binding_loss_fn = nn.BCEWithLogitsLoss()
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(device)

            binding_logits, token_logits = model(
                mirna=batch["mirna_input_ids"],
                mrna=batch["mrna_input_ids"],
                mrna_mask=batch["mrna_attention_mask"],
                mirna_mask=batch["mirna_attention_mask"]
            )

            binding_loss = binding_loss_fn(binding_logits.squeeze(-1), batch["binding_labels"].view(-1).float())
            token_loss = token_loss_fn(token_logits.view(-1, 3), batch["labels"].view(-1))

            loss = binding_loss + token_loss
            total_loss += loss.item()

            binding_preds = (torch.sigmoid(binding_logits) > 0.5).long()
            all_binding_preds.extend(binding_preds.cpu().numpy().flatten())
            all_binding_labels.extend(batch["binding_labels"].cpu().numpy().flatten())

            # Get argmax predictions
            token_preds = torch.argmax(token_logits, dim=-1)  # (B, L)
            # Convert all B (1) to I (2)
            token_preds = torch.where(token_preds == 1, torch.tensor(2, device=token_preds.device), token_preds)
            
            # Convert logits to probabilities using softmax for AUROC/AUPRC
            token_probs = F.softmax(token_logits, dim=-1)  # (B, L, 3)
            # Probability of being B or I (positive class for span detection)
            # Sum probabilities of class 1 (B) and class 2 (I)
            positive_probs = token_probs[:, :, 1] + token_probs[:, :, 2]  # (B, L)
            
            all_token_probs.extend(positive_probs.cpu().numpy().flatten())
            all_token_preds.extend(token_preds.cpu().numpy().flatten())
            all_token_labels.extend(batch["labels"].cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader)
    print("dataset size: ",len(dataloader))
    binding_accuracy = (np.array(all_binding_preds) == np.array(all_binding_labels)).mean()
    
    # Filter out the ignored index (-100) for token accuracy calculation
    all_token_labels = np.array(all_token_labels)
    all_token_probs = np.array(all_token_probs)
    all_token_preds = np.array(all_token_preds)
    valid_indices = all_token_labels != -100

    # Convert multi-class labels to binary: O (0) -> 0, B (1) or I (2) -> 1
    all_labels_binary = (all_token_labels[valid_indices] > 0).astype(int)
    # Convert predictions to binary: O (0) -> 0, I (2) -> 1 (B already converted to I)
    all_preds_binary = (all_token_preds[valid_indices] > 0).astype(int)
    all_probs = all_token_probs[valid_indices]
    
    # Filter out cases where all labels are 0 (no positive examples in sequence)
    if len(all_labels_binary) == 0 or all_labels_binary.sum() == 0:
        print("Warning: No positive labels found. Cannot calculate AUROC.")
        return None
    
    try:
        auroc = roc_auc_score(all_labels_binary, all_probs)
        auprc = average_precision_score(all_labels_binary, all_probs)
        # Use binary predictions directly (no threshold needed)
        precision = precision_score(all_labels_binary, all_preds_binary)
        recall = recall_score(all_labels_binary, all_preds_binary)
        f1 = f1_score(all_labels_binary, all_preds_binary)
        return {"AUROC": auroc, 
        "AUPRC": auprc, 
        "Precision": precision, 
        "Recall": recall, 
        "F1": f1}
    except ValueError as e:
        print(f"Error calculating AUROC: {e}")
        print(f"Unique labels: {np.unique(all_labels_binary)}")
        print(f"Probabilities range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
        return None


def main():
    # Configuration
    ckpt_path = "/home/mcb/users/jgu13/projects/mirLM/checkpoints/TargetScan/TokenClassification/520/best_accuracy_0.9948_epoch23.pth"
    data_path = os.path.join(PROJ_HOME, "TargetScan_dataset/TargetScan_validation_500_multiseeds_random_samples.csv")
    
    mrna_max_len = 520
    mirna_max_len = 24
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    D_val = load_dataset(data_path, sep=',', parse_seeds=True)
    
    tokenizer = CharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],
        model_max_length=mrna_max_len,
        padding_side="right"
    )
    
    ds_val = TokenClassificationDataset(
        df=D_val,
        tokenizer=tokenizer,
        mrna_max_len=mrna_max_len,
        mirna_max_len=mirna_max_len
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
        epochs=1,
        embed_dim=1024,
        ff_dim=2048,
        batch_size=32,
        lr=3e-5,
        seed=54,
        predict_span=False,
        predict_binding=True,
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
    metrics = evaluate_span_metrics(model, val_loader, device)
    
    if metrics is not None:
        print("=" * 60)
        print("BIO Token Classification metrics:")
        print(f"AUROC: {metrics['AUROC']:.4f}")
        print(f"AUPRC: {metrics['AUPRC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1: {metrics['F1']:.4f}")
        print("=" * 60)
        # save the auroc score to a file
        with open("span_auroc.txt", "w") as f:
            f.write(f"AUROC: {metrics['AUROC']:.4f}\n")
            f.write(f"AUPRC: {metrics['AUPRC']:.4f}\n")
            f.write(f"Precision: {metrics['Precision']:.4f}\n")
            f.write(f"Recall: {metrics['Recall']:.4f}\n")
            f.write(f"F1: {metrics['F1']:.4f}")
    else:
        print("\nFailed to calculate metrics.")


if __name__ == "__main__":
    main()

