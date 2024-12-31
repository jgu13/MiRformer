import os
import json
import time
import torch
import random
import argparse
import numpy as np
from torch import nn, optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from Hyena_layer import HyenaDNAPreTrainedModel, HyenaDNAModel
from Data_pipeline import CharacterTokenizer, miRawDataset

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

def evaluate(model, device, test_loader, mRNA_max_length, miRNA_max_length):
    """Test loop."""
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for seq, seq_mask, target in test_loader:
            seq, target, seq_mask = (
                seq.to(device),
                target.to(device),
                seq_mask.to(device),
            )
            output = model(
                device=device,
                input_ids=seq,
                input_mask=seq_mask,
                max_mRNA_length=mRNA_max_length,
                max_miRNA_length=miRNA_max_length,
            )
            probabilities = torch.sigmoid(output.squeeze()).cpu().numpy().tolist()
            targets = target.cpu().view(-1).numpy().tolist()
            predictions.extend(probabilities)
            true_labels.extend(targets)

    return predictions, true_labels

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn, if reproducibility is needed:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate MLP model on test dataset.")
    parser.add_argument(
        "--mRNA_max_len",
        type=int,
        default=1000,
        help="Maximum length of mRNA sequence used for training (default: 1000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: auto-detected)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size to load test data (default: 16)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/mcb/users/jgu13/projects/mirLM/data/training_1000.csv",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to checkpoint"
    )
    args = parser.parse_args()

    # Extract arguments
    mRNA_max_length = args.mRNA_max_len
    device = args.device
    batch_size=args.batch_size
    test_dataset_path=args.dataset_path
    ckpt_path=args.ckpt_path

    # Other fixed parameters
    model_name = "HyenaDNA"
    test_dataset_name = os.path.basename(test_dataset_path).split('.')[0] 
    miRNA_max_length = 24
    backbone_cfg = None
    use_head=True
    n_classes=1

    print("Evaluation -- Start --")
    print("mRNA length:", mRNA_max_length)
    print("Using device:", device)

    # Load pretrained model
    pretrained_model_name="hyenadna-small-32k-seqlen"
    path = f"{PROJ_HOME}/checkpoints"
    
    seed_everything(seed=42)
    
    # use the pretrained Huggingface wrapper instead
    model = HyenaDNAPreTrainedModel.from_pretrained(
        path,
        pretrained_model_name,
        download=False,
        config=backbone_cfg,
        device=device,
        use_head=use_head,
        n_classes=n_classes,
    )
    loaded_data = torch.load(ckpt_path)
    model.load_state_dict(loaded_data["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")
    model.to(device)
    
    test_dataset = pd.read_csv(test_dataset_path)
    assert all(col in test_dataset.columns for col in ['miRNA sequence', 'mRNA sequence']), "column names must contain 'mRNA sequence' and 'miRNA sequence'."
    max_length = miRNA_max_length + mRNA_max_length + 2
    
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "U", "N"],  # add RNA characters, N is uncertain
        model_max_length=max_length,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )
    
    ds_test = miRawDataset(
        test_dataset,
        mRNA_max_length=mRNA_max_length,
        miRNA_max_length=miRNA_max_length,
        tokenizer=tokenizer,
        use_padding=True,
        rc_aug=False,
        add_eos=False,
    )
    
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    
    predictions, true_labels = evaluate(
        model=model,
        device=device,
        test_loader=test_loader,
        mRNA_max_length=mRNA_max_length,
        miRNA_max_length=miRNA_max_length
    )

    # Save predictions and true labels
    output_path = os.path.join(PROJ_HOME, "Performance", test_dataset_name, model_name)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f"predictions_{mRNA_max_length}.json"), "w") as f:
        json.dump({"predictions": predictions, "true_labels": true_labels}, f)

    print("Evaluation completed. Predictions saved to", output_path)

if __name__ == "__main__":
    main()
    