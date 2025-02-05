import os
import json
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from torch import distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split

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
    def __init__(self, 
                 dataset, 
                 mRNA_max_length=40, 
                 miRNA_max_length=26,
                 mRNA_col="mRNA sequence",
                 miRNA_col="miRNA sequence"):
        self.mRNA_max_length = mRNA_max_length
        self.miRNA_max_length = miRNA_max_length
        self.data = dataset
        self.mRNA_sequences = dataset[[mRNA_col]].values
        self.miRNA_sequences = dataset[[miRNA_col]].values
        self.labels = dataset[["label"]].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert data to PyTorch tensors
        mRNA_seq = self.mRNA_sequences[idx][0]
        miRNA_seq = self.miRNA_sequences[idx][0]
        labels = self.labels[idx]

        mRNA_seq = mRNA_seq.replace("U", "T")
        miRNA_seq = miRNA_seq.replace("U", "T")
        
        # pad to max length
        if len(mRNA_seq) < self.mRNA_max_length:
            num_pad = self.mRNA_max_length - len(mRNA_seq)
            # pad with N
            mRNA_seq = "".join(list(mRNA_seq) + list("N" * num_pad))
        if len(miRNA_seq) < self.miRNA_max_length:
            num_pad = self.miRNA_max_length - len(miRNA_seq)
            # pad with N
            miRNA_seq = "".join(list(miRNA_seq) + list("N" * num_pad))

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


# load and sequence data
def load_data(dataset=None, sep=","):
    if dataset:
        # if dataset is a path to a file
        if isinstance(dataset, str):
            if (
                dataset.endswith(".csv")
                or dataset.endswith(".txt")
                or dataset.endswith(".tsv")
            ):
                D = pd.read_csv(dataset, sep=sep)
            elif dataset.endswith(".xlsx"):
                D = pd.read_excel(dataset, sep=sep)
            elif dataset.endswith(".json"):
                with open(dataset, "r") as f:
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


def train_model(
    model: BaselineCNN, 
    train_loader: DataLoader, 
    loss_fn: nn.BCELoss, 
    optimizer: optim.AdamW, 
    device: torch.device, 
    epoch: int, 
    accumulation_step: int = None,
    ddp: bool = False
):
    model.train()
    epoch_loss = 0.0
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
            if (batch_idx + 1) % accumulation_step == 0:
                # backward pass
                optimizer.step()
                optimizer.zero_grad()
                if ddp:
                    print(
                            f"[Rank {dist.get_rank()}] "
                            f"Train Epoch: {epoch} "
                            f"[{(batch_idx+1) * len(sequences)}/{len(train_loader.sampler)} "
                            f"({100.0 * (batch_idx+1) / len(train_loader):.0f}%)] "
                            f"Loss: {loss.item():.6f}\n",
                            flush=True
                        )
                else:
                    print(
                            f"Train Epoch: {epoch} "
                            f"[{(batch_idx+1) * len(sequences)}/{len(train_loader.dataset)} "
                            f"({100.0 * (batch_idx+1) / len(train_loader):.0f}%)] "
                            f"Loss: {loss.item():.6f}\n",
                            flush=True
                        )
        else:
            loss.backward()
            optimizer.step()
            print(
                    f"[Rank {dist.get_rank()}] "
                    f"Train Epoch: {epoch} "
                    f"[{(batch_idx+1) * len(sequences)}/{len(train_loader.sampler)} "
                    f"({100.0 * (batch_idx+1) / len(train_loader):.0f}%)] "
                    f"Loss: {loss.item():.6f}\n"
                )
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    return avg_loss

def test(model, 
         test_loader, 
         device):
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

    print(
        "\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(
            correct, len(test_loader.dataset), accuracy
        )
    )
    return accuracy

def test_ddp(model, test_loader, device):
    model.eval()
    local_correct = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            probabilities = torch.sigmoid(outputs.squeeze())
            predictions = (probabilities > 0.5).long()
            local_correct += predictions.eq(labels.squeeze()).sum().item()

    # convert to gpu tensor
    correct_tensor = torch.tensor(local_correct, dtype=torch.long, device=device)
    # Sum across all ranks
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    # Get all correct counts
    global_correct = correct_tensor.item()
    # compute global accuracy
    global_accuracy = 100.0 * global_correct / len(test_loader.dataset)
    if dist.get_rank() == 0:
        print(
            f"Test set: Accuracy: {global_correct}/{len(test_loader.dataset)} "
            f"({global_accuracy:.2f}%)\n"
        )
    return global_accuracy

def evaluate(model, 
            test_loader, 
            device):
    """
    Evaluate model performance on test dataset
    """
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

def save_checkpoint(model, optimizer, epoch, average_loss, accuracy, path):
    """
    Helper function for saving model/optimizer state dicts.
    Only rank 0 should save to avoid file corruption.
    """
    # If wrapped in DDP, actual parameters live in model.module
    model_state_dict = model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": average_loss,
        "accuracy": accuracy,
    }, path)

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
    parser = argparse.ArgumentParser(description="Run training script for binary classification.")
    parser.add_argument(
        "--mRNA_max_len",
        type=int,
        default=1000,
        help="Maximum length of mRNA sequences (default: 1000)",
    )
    parser.add_argument(
        "--miRNA_max_len",
        type=int,
        default=28,
        help="Maximum length of mRNA sequences (default: 28)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run training on (default: auto-detected)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size to load dataset"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="training data",
        help="Name of the folder to save training performance and checkpoints"
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="path/to/training/data",
        help="Path to training data"
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default="path/to/validation/data",
        help="Path to validation data"
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="path/to/test/data",
        help="Path to test data"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model on test dataset"
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use DistributedDataParallel for multi-gpu training"
    )
    parser.add_argument(
        "--resume_ckpt", 
        type=str, 
        default=None, 
        help="Path to checkpoint to resume from."
    )
    args = parser.parse_args()

    # Extract arguments
    mRNA_max_length = args.mRNA_max_len
    miRNA_max_length = args.miRNA_max_len
    device = torch.device(args.device)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    train_dataset_path = args.train_dataset_path
    val_dataset_path = args.val_dataset_path
    test_dataset_path = args.test_dataset_path
    evaluate_flag = args.evaluate
    ddp_flag = args.ddp
    resume_ckpt = args.resume_ckpt
    
    # Other fixed arguments
    accumulation_step = 256 // batch_size
    model_name = "CNN"
    
    # initialize process group if DDP is used
    if ddp_flag:
        seed_everything(seed=42)
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{local_rank}" if (torch.cuda.is_available() and ddp_flag) else "cpu"
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        
    if rank == 0:
        print("------ CNN Binary Classification starts here------")
        print(f"mRNA length = {mRNA_max_length}")
        print(f"On device {device}")
        print(f"For {num_epochs} epochs")
        print(f"Using device {device}")
        print(f"DDP = {ddp_flag}, World size = {world_size}, Rank = {rank}")
        print(f"Resume from checkpoint path {resume_ckpt}")
    
    config_path = os.path.join(PROJ_HOME, "checkpoints", "CNN", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    lr = config["learning_rate"] * world_size
    weight_decay = config["weight_decay"]

    if evaluate_flag:
        D_test = load_data(dataset=test_dataset_path)
        test_dataset = BaselineCNNDataset(
            dataset=D_test,
            mRNA_max_length=mRNA_max_length,
            miRNA_max_length=miRNA_max_length,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        # load and preprocess data
        D_train = load_data(dataset=train_dataset_path)
        D_test = load_data(dataset=val_dataset_path)
        # max_mRNA_len = max([len(seq) for seq in D["mRNA sequence"]])
        # print("Max mRNA sequence len = ", max_mRNA_len)
        # max_miRNA_len = max([len(seq) for seq in D["miRNA sequence"]])
        # print("Max miRNA sequence len = ", max_miRNA_len)

        # dataloaders
        train_dataset = BaselineCNNDataset(
            dataset=D_train,
            mRNA_max_length=mRNA_max_length,
            miRNA_max_length=miRNA_max_length,
            mRNA_col="mRNA sequence",
        )
        test_dataset = BaselineCNNDataset(
            dataset=D_test,
            mRNA_max_length=mRNA_max_length,
            miRNA_max_length=miRNA_max_length,
            mRNA_col="mRNA sequence"
        )
        
        # Distributed Samplers
        if ddp_flag:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            shuffle_data = False  # Sampler does the shuffling for train
        else:
            train_sampler = None
            test_sampler = None
            shuffle_data = True
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)

    # init model
    model = BaselineCNN(
        input_size=mRNA_max_length + miRNA_max_length,
        num_classes=config["num_classes"],
        kernel_size=config["kernel_size"]
    ).to(device)  # input_size = padded miRNA length + padded mRNA length

    if evaluate_flag:
        ckpt_path = os.path.join(PROJ_HOME, "checkpoints", dataset_name, "CNN", str(mRNA_max_length), "checkpoint_epoch_final.pth")
        loaded_data = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(loaded_data['model_state_dict'])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        # DPP wrapper
        if ddp_flag:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
        
        loss_fn = nn.BCELoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )

        train_loss_list = []
        test_accuracy_list = []
        
    model.to(device)
    
    # evaluation only
    if evaluate_flag:
        test_dataset_name = os.path.basename(test_dataset_path).split('.')[0]
        predictions, true_labels = evaluate(
            model=model,
            test_loader=test_loader,
            device=device
        )
        # Save predictions and true labels
        output_path = os.path.join(PROJ_HOME, "Performance", test_dataset_name, model_name)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f"predictions_{mRNA_max_length}.json"), "w") as f:
            json.dump({"predictions": predictions, "true_labels": true_labels}, f)

        print("Evaluation completed. Predictions saved to", output_path)
    else:
        # train model
        model_checkpoint_dir = os.path.join(PROJ_HOME, "checkpoints", dataset_name, "CNN", str(mRNA_max_length))
        os.makedirs(model_checkpoint_dir, exist_ok=True)
        
        start = time.time()
        for epoch in range(num_epochs):
            if ddp_flag:
                # Important: set epoch for DistributedSampler to shuffle data consistently
                train_loader.sampler.set_epoch(epoch)
                
            train_loss = train_model(
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                accumulation_step=accumulation_step,
                ddp=ddp_flag
            )
            if ddp_flag:
                accuracy = test_ddp(model=model, 
                                    test_loader=test_loader, 
                                    device=device)
            else:
                accuracy = test(model=model, 
                                test_loader=test_loader, 
                                device=device)
            if rank == 0:
                train_loss_list.append(train_loss)
                test_accuracy_list.append(accuracy)
                cost = time.time() - start
                remain = (cost / (epoch + 1)) * (num_epochs - epoch - 1) /60 /60
                print(f"Remaining: {remain} hours")
            
            if epoch % 10 == 0 and rank == 0:
                save_checkpoint(model=model,
                                optimizer=optimizer,
                                epoch=epoch,
                                average_loss=train_loss,
                                accuracy=accuracy,
                                path=os.path.join(model_checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))

        if rank == 0:
            # save final model
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            average_loss=train_loss,
                            accuracy=accuracy,
                            path=os.path.join(model_checkpoint_dir, f"checkpoint_epoch_final.pth"))
            
            time_taken = time.time() - start
            print("Time taken for {} epoch = {} min.".format(num_epochs, time_taken / 60))

            # save test_accuracy
            perf_dir = os.path.join(PROJ_HOME, "Performance", dataset_name, model_name)
            os.makedirs(perf_dir, exist_ok=True)
            with open(
                os.path.join(perf_dir, f"evaluation_accuracy_{mRNA_max_length}.json"), "w"
            ) as fp:
                json.dump(test_accuracy_list, fp)
            with open(
                os.path.join(perf_dir, f"train_loss_{mRNA_max_length}.json"), "w"
            ) as fp:
                json.dump(train_loss_list, fp)
    
        # destroy process group to clean up
        if ddp_flag:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
