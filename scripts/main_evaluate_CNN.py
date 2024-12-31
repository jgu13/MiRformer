import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
# from baseline_CNN import BaselineCNN, BaselineCNNDataset

PROJ_HOME = os.path.join(os.path.expanduser("~/projects/mirLM"))

class BaselineCNN(nn.Module):
    '''
    CNN model to predict sequence classification
    '''
    def __init__(self, 
                 input_size: int, 
                 num_classes: int, 
                 kernel_size: int = 5):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv1d(5, 10, kernel_size=kernel_size, padding=0)
        self.fc1 = nn.Linear(10 * (input_size - kernel_size + 1), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Forward pass to train CNN
        '''
        x = self.conv1(x)  # (batch_size, C_out, L_out)
        x = x.view(x.size(0), -1)  # (batch_size, C_out * L_out)
        x = self.relu(self.fc1(x))  # (batch_size, 100)
        x = self.bn1(x)  # (batch_size, 100)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return x


# nucleotide dataset class
class BaselineCNNDataset(Dataset):
    def __init__(self, dataset, mRNA_max_length=40, miRNA_max_length=26):
        self.mRNA_max_length = mRNA_max_length
        self.miRNA_max_length = miRNA_max_length
        self.data = dataset
        self.mRNA_sequences = dataset[["mRNA sequence"]].values
        self.miRNA_sequences = dataset[["miRNA sequence"]].values
        self.labels = dataset[["label"]].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert data to PyTorch tensors
        mRNA_seq = self.mRNA_sequences[idx][0]
        miRNA_seq = self.miRNA_sequences[idx][0]
        labels = self.labels[idx]

        # pad to max length
        if len(mRNA_seq) < self.mRNA_max_length:
            num_pad = self.mRNA_max_length - len(mRNA_seq)
            # pad with N
            mRNA_seq = "".join(list(mRNA_seq) + list("N" * num_pad))
        # truncate to max length
        elif len(mRNA_seq) > self.mRNA_max_length:
            mRNA_seq = ''.join(list(mRNA_seq)[:self.mRNA_max_length])
        # pad to max length
        if len(miRNA_seq) < self.miRNA_max_length:
            num_pad = self.miRNA_max_length - len(miRNA_seq)
            # pad with N
            miRNA_seq = "".join(list(miRNA_seq) + list("N" * num_pad))
        # truncate to max length
        elif len(miRNA_seq) > self.miRNA_max_length:
            miRNA_seq = ''.join(list(miRNA_seq)[:self.miRNA_max_length])

        concat_seq = miRNA_seq + mRNA_seq
        # print("length of concat sequence = ", len(concat_seq))

        # to torch tensor
        encoded_sequences = np.asarray(encode_dna(concat_seq))  # (seq_len, C_in)
        # print("encoded RNA sequence length = ", encoded_sequences.shape)
        encoded_sequences = torch.FloatTensor(encoded_sequences).permute(
            1, 0
        )  # (C_in, seq_len)
        # print("RNA tensor shape = ", encoded_sequences.size())
        labels = torch.FloatTensor(labels)

        return encoded_sequences, labels


# one-hot DNA encoder
def encode_dna(seq):
    encoding = {
        "A": [1, 0, 0, 0, 0],
        "C": [0, 1, 0, 0, 0],
        "G": [0, 0, 1, 0, 0],
        "T": [0, 0, 0, 1, 0],
        "U": [0, 0, 0, 0, 1],
        "N": [1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4],
    }

    return [encoding.get(base, [0, 0, 0, 0, 0]) for base in seq]

def evaluate(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            probabilities = torch.sigmoid(outputs.squeeze()).cpu().numpy().tolist()
            labels = labels.cpu().view(-1).numpy().tolist()
            predictions.extend(probabilities)
            true_labels.extend(labels)
    return predictions, true_labels

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn, if reproducibility is needed:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_argument_parser():
    input_parser = argparse.ArgumentParser(description="Run training script for binary classification.")
    input_parser.add_argument(
        "--mRNA_max_len",
        type=int,
        default=1000,
        help="Maximum length of mRNA sequences (default: 1000)",
    )
    input_parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run training on (default: auto-detected)",
    )
    input_parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size to load dataset"
    )
    input_parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/mcb/users/jgu13/projects/mirLM/data/training_1000.csv",
        help="Path to test dataset"
    )
    input_parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default=None, 
        help="Path to checkpoint."
    )
    return input_parser

def main():
    # Parse command-line arguments
    input_parser = get_argument_parser()
    args = input_parser.parse_args()

    # Extract arguments
    mRNA_max_length = args.mRNA_max_len
    device = torch.device(args.device)
    print("On device = ", device)
    batch_size = args.batch_size
    test_dataset_path = args.dataset_path
    ckpt_path = args.ckpt_path
    
    # Other fixed parameters 
    miRNA_max_length = 28
    model_name = "CNN"
    test_dataset_name = os.path.basename(test_dataset_path).split('.')[0]

    # load and preprocess data
    test_dataset = pd.read_csv(test_dataset_path)
    assert all(col in test_dataset.columns for col in ['miRNA sequence', 'mRNA sequence']), "column names must contain 'mRNA sequence' and 'miRNA sequence'."
    
    seed_everything(seed=42)
    
    test_dataset = BaselineCNNDataset(
        dataset=test_dataset,
        mRNA_max_length=mRNA_max_length, 
        miRNA_max_length=miRNA_max_length,
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # init model
    config_path = os.path.join(PROJ_HOME, "checkpoints", model_name, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = BaselineCNN(
        input_size=mRNA_max_length + miRNA_max_length,
        num_classes=config["num_classes"],
        kernel_size=config["kernel_size"]
    )
    loaded_data = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(loaded_data['model_state_dict'])
    print(f"Loaded checkpoint from {ckpt_path}")
    model.to(device)
    
    # evaluation
    predictions, true_labels = evaluate(
        model=model,
        test_loader=test_loader,
        device=device)
    
    # Save predictions and true labels
    output_path = os.path.join(PROJ_HOME, "Performance", test_dataset_name, model_name)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f"predictions_{mRNA_max_length}.json"), "w") as f:
        json.dump({"predictions": predictions, "true_labels": true_labels}, f)

    print("Evaluation completed. Predictions saved to", output_path)

if __name__ == '__main__':
    main()
    
    