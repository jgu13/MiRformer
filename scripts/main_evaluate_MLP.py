import os
import json
import math
import torch
import random
import argparse
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pandas as pd

# Local imports
from Hyena_layer import HyenaDNAPreTrainedModel,LinearHead
from Data_pipeline import CharacterTokenizer, CustomDataset

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

def compute_cross_attention(Q: torch.tensor, 
                            K: torch.tensor, 
                            V: torch.tensor, 
                            Q_mask: torch.tensor, 
                            K_mask: torch.tensor):
    '''
    Compute cross entropy
    '''
    d_model = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
        d_model
    )  # [batchsize, mRNA_seq_len, miRNA_seq_len]
    # expand K mask to mask out keys, make each query only attend to valid keys
    K_mask = K_mask.unsqueeze(1).expand(
        -1, Q.shape[1], -1
    )  # [batchsize, mRNA_seq_len, miRNA_seq_len]
    scores = scores.masked_fill(K_mask == 0, -1e9)
    # apply softmax on the key dimension
    attn_weights = F.softmax(scores, dim=-1)  # [batchsize, mRNA_seq_len, miRNA_seq_len]
    cross_attn = torch.matmul(attn_weights, V)  # [batchsize, mRNA_seq_len, d_model]
    # expand Q mask to mask out queries, zero out padded queries
    valid_counts = Q_mask.sum(dim=1, keepdim=True)  # [batchsize, 1]
    Q_mask = Q_mask.unsqueeze(-1).expand(
        -1, -1, d_model
    )  # [batchsize, mRNA_seq_len, d_model]
    cross_attn = cross_attn * Q_mask
    # average pool over seq_length
    cross_attn = cross_attn.sum(dim=1) / valid_counts  # [batchsize, d_model]
    return cross_attn

def evaluate(HyenaDNA_feature_extractor, 
             MLP_head,
             Q_layer,
             KV_layer, 
             device, 
             test_loader):
    """Evaluation loop."""
    MLP_head.eval()
    HyenaDNA_feature_extractor.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target in test_loader:
            mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target = (
                mRNA_seq.to(device),
                miRNA_seq.to(device),
                mRNA_seq_mask.to(device),
                miRNA_seq_mask.to(device),
                target.to(device),
            )
            mRNA_hidden_states = HyenaDNA_feature_extractor(
                device=device, input_ids=mRNA_seq, use_only_miRNA=False
            )
            miRNA_hidden_states = HyenaDNA_feature_extractor(
                device=device, input_ids=miRNA_seq, use_only_miRNA=False
            )
            Q = Q_layer(miRNA_hidden_states)  # [batchsize, mRNA_seq_len, d_model]
            K = KV_layer(mRNA_hidden_states)  # [batchsize, miRNA_seq_len, d_model]
            V = KV_layer(mRNA_hidden_states)  # [batchsize, miRNA_seq_len, d_model]
            Q_mask = mRNA_seq_mask
            K_mask = miRNA_seq_mask
            cross_attn = compute_cross_attention(
                Q=Q,
                K=K, 
                V=V, 
                Q_mask=Q_mask, 
                K_mask=K_mask
            )
            output = MLP_head(cross_attn)  # (batch_size, 1)
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
        "--dataset_name",
        type=str,
        default="train dataset",
        help="The name of the folder that indicate which training dataset is used"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/mcb/users/jgu13/projects/mirLM/data/training_1000.csv",
        help="Path to test dataset"
    )
    args = parser.parse_args()

    # Extract arguments
    mRNA_max_length = args.mRNA_max_len
    device = args.device
    batch_size=args.batch_size
    dataset_name=args.dataset_name
    test_dataset_path=args.dataset_path

    # Other fixed parameters
    miRNA_max_length = 26
    model_name = "MLP"
    test_dataset_name = os.path.basename(test_dataset_path).split('.')[0]
    backbone_cfg = None
    download = False
    use_head=False
    n_classes=1

    print("Evaluation -- Start --")
    print("mRNA length:", mRNA_max_length)
    print("Using device:", device)

    seed_everything(42)
    
    # Load pretrained model
    pretrained_model_name="hyenadna-small-32k-seqlen"
    path = f"{PROJ_HOME}/checkpoints"
    # use the pretrained Huggingface wrapper instead
    HyenaDNA_feature_extractor = HyenaDNAPreTrainedModel.from_pretrained(
        path,
        pretrained_model_name,
        download=download,
        config=backbone_cfg,
        device=device,
        use_head=use_head,
    )
    # first check if it is a local path
    pretrained_model_name_or_path = os.path.join(path, pretrained_model_name)
    if os.path.isdir(pretrained_model_name_or_path):
        if backbone_cfg is None:
            with open(
                os.path.join(pretrained_model_name_or_path, "config.json"),
                "r",
                encoding="utf-8",
            ) as f:
                backbone_cfg = json.load(f)
        elif isinstance(backbone_cfg, str) and backbone_cfg.endswith(".json"):
            with open(backbone_cfg, "r", encoding="utf-8") as f:
                backbone_cfg = json.load(f)
        else:
            assert isinstance(
                backbone_cfg, dict
            ), "self-defined backbone config must be a dictionary."

    hidden_sizes = [backbone_cfg["d_model"] * 2, backbone_cfg["d_model"] * 2]
    # first check if it is a local path
    MLP_head = LinearHead(
        d_model=backbone_cfg["d_model"], d_output=n_classes, hidden_sizes=hidden_sizes
    )
    Q_layer = nn.Linear(backbone_cfg["d_model"], backbone_cfg["d_model"])
    KV_layer = nn.Linear(backbone_cfg["d_model"], backbone_cfg["d_model"])
    
    # load checkpoint
    ckpt_path = os.path.join(PROJ_HOME, "checkpoints", dataset_name, "MLP", "checkpoint_epoch_final.pth")
    loaded_data = torch.load(ckpt_path, map_location=device)
    print("Loaded checkpoint from ", ckpt_path)
    MLP_head.load_state_dict(loaded_data["model_state_dict"])
    Q_layer.load_state_dict(loaded_data["Q_layer_state_dict"])
    KV_layer.load_state_dict(loaded_data["KV_layer_state_dict"])
    
    HyenaDNA_feature_extractor.to(device)
    MLP_head.to(device)
    Q_layer.to(device)
    KV_layer.to(device)

    D = pd.read_csv(test_dataset_path)
    assert all(col in D.columns for col in ['miRNA sequence', 'mRNA sequence']), "column names must contain 'mRNA sequence' and 'miRNA sequence'."
    
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=mRNA_max_length + miRNA_max_length + 2,
        add_special_tokens=False,
        padding_side="left",
    )
    
    ds_test = CustomDataset(
        D,
        mRNA_max_length=mRNA_max_length, # pad to mRNA max length
        miRNA_max_length=miRNA_max_length, # pad to miRNA max length
        tokenizer=tokenizer,
        use_padding=True,
        rc_aug=False,
        add_eos=True,
    )

    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    predictions, true_labels = evaluate(
        HyenaDNA_feature_extractor=HyenaDNA_feature_extractor,
        MLP_head=MLP_head,
        Q_layer=Q_layer,
        KV_layer=KV_layer,
        device=device,
        test_loader=test_loader,
    )

    # Save predictions and true labels
    output_path = os.path.join(PROJ_HOME, "Performance", test_dataset_name, model_name)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f"predictions_{mRNA_max_length}.json"), "w") as f:
        json.dump({"predictions": predictions, "true_labels": true_labels}, f)

    print("Evaluation completed. Predictions saved to", output_path)

if __name__ == "__main__":
    main()
