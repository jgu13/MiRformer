import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import json
import time


class BaselineCNN(nn.Module):
    def __init__(self, input_size, num_classes, kernel_size=5):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv1d(5, 10, kernel_size=kernel_size, padding=0)
        self.fc1 = nn.Linear(10 * (input_size - kernel_size + 1), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x) # (batch_size, C_out, L_out)
        x = x.view(x.size(0), -1) # (batch_size, C_out * L_out)
        x = self.relu(self.fc1(x)) # (batch_size, 100)
        x = self.bn1(x) # (batch_size, 100)
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
            mRNA_seq = ''.join(list(mRNA_seq) + list('N'*num_pad))
        if len(miRNA_seq) < self.miRNA_max_length:
            num_pad = self.miRNA_max_length - len(miRNA_seq)
            # pad with N
            miRNA_seq = ''.join(list(miRNA_seq) + list('N'*num_pad))
        
        concat_seq = miRNA_seq + mRNA_seq
        # print("length of concat sequence = ", len(concat_seq))
        
        # to torch tensor
        encoded_sequences = np.asarray(encode_dna(concat_seq)) # (seq_len, C_in)
        # print("encoded RNA sequence length = ", encoded_sequences.shape)
        encoded_sequences = torch.FloatTensor(encoded_sequences).permute(1, 0) # (C_in, seq_len)
        # print("RNA tensor shape = ", encoded_sequences.size())
        labels = torch.FloatTensor(labels)

        return encoded_sequences, labels


# one-hot DNA encoder
def encode_dna(seq):
    encoding = {'A': [1,0,0,0,0], 
                'C': [0,1,0,0,0], 
                'G': [0,0,1,0,0], 
                'T': [0,0,0,1,0],
                'U': [0,0,0,0,1],
                'N': [1/4,1/4,1/4,1/4,1/4]}
    
    return [encoding.get(base, [0,0,0,0,0]) for base in seq]


# load and sequence data
def load_data(dataset=None, sep=','):
    if dataset:
        # if dataset is a path to a file
        if isinstance(dataset, str):
            if dataset.endswith('.csv') or dataset.endswith('.txt') or dataset.endswith('.tsv'):
                D = pd.read_csv(dataset, sep=sep)
            elif dataset.endswith('.xlsx'):
                D = pd.read_excel(dataset, sep=sep)
            elif dataset.endswith('.json'):
                with open(dataset, 'r') as f:
                    D = json.load(f)
            else:
                print(f"Unrecognized format of {dataset}")
                D = None
        # if dataset is a pandas dataframe
        elif isinstance(dataset, pd.DataFrame):
            D = dataset
        else:
            print("Dataset must be a path or a pandas dataframe.")
            D = None
    else:
        print(f"Dataset is {dataset}.")
        D = None
    return D


def train_model(model, train_loader, loss_fn, optimizer, device, epoch, accumulation_step=None):
    model.train()
    epoch_loss=0.0
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # forward pass
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)

        loss = loss_fn(outputs.squeeze().sigmoid(), labels.squeeze())

        if accumulation_step:
            loss = loss / accumulation_step
            # calculate gradient
            loss.backward()
            if batch_idx % accumulation_step == 0:
                # backward pass
                optimizer.step()
                optimizer.zero_grad()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(sequences), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(sequences), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    return avg_loss

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            probabilities = torch.sigmoid(outputs.squeeze())
            predictions = (probabilities > 0.5).long()
            correct += predictions.eq(labels.squeeze()).sum().item()
    
    accuracy = 100 * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, len(test_loader.dataset), accuracy))
    return accuracy

def main():
    batch_size = 16
    num_epochs = 10
    num_classes = 1
    kernel_size = 12
    accumulation_step = 256 / batch_size
    mRNA_max_length = 5000
    miRNA_max_length = 28
    learning_rate = 3e-4
    weight_decay = 0.1
    
    dataset = "mirLM"
    model_name = "CNN"
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # load and preprocess data
    PROJ_HOME = os.path.join(os.path.expanduser("~/projects/mirLM"))
    D = load_data(dataset=os.path.join(PROJ_HOME, f"data/training_{mRNA_max_length}.csv"), sep=",")
    # D = D.sample(n=62654, random_state=34) # randomly sample 60k samples

    # split data
    D_train, D_test = train_test_split(D, test_size=0.2, random_state=34, shuffle=True)

    # dataloaders
    train_dataset = BaselineCNNDataset(dataset=D_train, mRNA_max_length=mRNA_max_length, miRNA_max_length=miRNA_max_length)
    test_dataset = BaselineCNNDataset(dataset=D_test, mRNA_max_length=mRNA_max_length, miRNA_max_length=miRNA_max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # init model
    model = BaselineCNN(input_size=mRNA_max_length + miRNA_max_length, num_classes=num_classes, kernel_size=kernel_size).to(device) # input_size = padded miRNA length + padded mRNA length
    loss_fn = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_loss_list = []
    test_accuracy_list = []
    # train model
    start = time.time()
    for epoch in range(num_epochs):
        train_loss = train_model(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, device=device, epoch=epoch,  accumulation_step=accumulation_step)
        accuracy = evaluate_model(model, test_loader, device)
        train_loss_list.append(train_loss)
        test_accuracy_list.append(accuracy)
    time_taken = time.time() - start
    print("Time taken for {} epoch = {} min.".format(num_epochs, time_taken/60))
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': accuracy,
            }, os.path.join(PROJ_HOME, 'checkpoints', 'mirLM', f'CNN', f'checkpoint_epoch_final.pth'))
    
    # save test_accuracy
    perf_dir = os.path.join(PROJ_HOME, "Performance", dataset, model_name)
    with open(os.path.join(perf_dir, f"test_accuracy_{mRNA_max_length}.json"), "w") as fp:
        json.dump(test_accuracy_list, fp)
    with open(os.path.join(perf_dir, f"train_loss_{mRNA_max_length}.json"), "w") as fp:
        json.dump(train_loss_list, fp)

main()