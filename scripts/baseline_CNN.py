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

# nucleotide dataset class
class BaselineCNNDataset(Dataset):
    def __init__(self, 
                 dataset, 
                 mRNA_max_length=40, 
                 miRNA_max_length=26,
                 add_linker=True,
                 mRNA_col="mRNA sequence",
                 miRNA_col="miRNA sequence"):
        self.mRNA_max_length = mRNA_max_length
        self.miRNA_max_length = miRNA_max_length
        self.data = dataset
        self.mRNA_sequences = dataset[[mRNA_col]].values
        self.miRNA_sequences = dataset[[miRNA_col]].values
        self.labels = dataset[["label"]].values
        self.add_linker = add_linker

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
        # elif len(mRNA_seq) >  self.mRNA_max_length:
        elif len(mRNA_seq) > self.mRNA_max_length:
            diff = len(mRNA_seq) - self.mRNA_max_length
            mRNA_seq = mRNA_seq[:-diff]
        if len(miRNA_seq) < self.miRNA_max_length:
            num_pad = self.miRNA_max_length - len(miRNA_seq)
            # pad with N
            miRNA_seq = "".join(list(miRNA_seq) + list("N" * num_pad))
        elif len(miRNA_seq) > self.miRNA_max_length:
            diff = len(miRNA_seq) - self.miRNA_max_length
            miRNA_seq = miRNA_seq[:-diff] # truncate in the 5' end 
        
        mRNA_seq = mRNA_seq.replace("U", "T")
        miRNA_seq = miRNA_seq.replace("U", "T")[::-1]

        if self.add_linker:
            concat_seq = miRNA_seq + "NNNNNN" + mRNA_seq
        else:
            concat_seq = miRNA_seq + mRNA_seq
        # print("length of concat sequence = ", len(concat_seq))

        # to torch tensor
        encoded_sequences = np.asarray(BaselineCNN.encode_dna(concat_seq))  # (seq_len, C_in)
        # print("encoded RNA sequence length = ", encoded_sequences.shape)
        encoded_sequences = torch.FloatTensor(encoded_sequences).permute(
            1, 0
        )  # (C_in, seq_len)
        # print("RNA tensor shape = ", encoded_sequences.size())
        labels = torch.FloatTensor(labels)

        return encoded_sequences, labels


class BaselineCNN(nn.Module):
    """
    CNN model to predict sequence classification and encapsulate the training pipeline.
    """

    def __init__(self, 
                 input_size: int, 
                 num_classes: int, 
                 kernel_size: int = 5):
        """
        Initialize the CNN model.
        """
        super(BaselineCNN, self).__init__()
        # Define the model layers
        self.conv1 = nn.Conv1d(5, 10, kernel_size=kernel_size, padding=0)
        self.fc1 = nn.Linear(10 * (input_size - kernel_size + 1), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        # Store architecture parameters if needed later
        self.input_size = input_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Forward pass.
        """
        x = self.conv1(x)  # (batch_size, C_out, L_out)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return x

    # -----------------------
    # Training and Evaluation
    # -----------------------

    def train_model(self, model, train_loader, loss_fn, optimizer, device, epoch, accumulation_step=None, ddp=False):
        """
        Train the model for one epoch.
        The parameter `model` is the instance (or its DDP wrapper) used for the forward pass.
        """
        model.train()
        epoch_loss = 0.0
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = loss_fn(outputs.squeeze().sigmoid(), labels.squeeze())

            if accumulation_step:
                loss = loss / accumulation_step
                loss.backward()
                if (batch_idx + 1) % accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if ddp:
                        print(
                            f"[Rank {dist.get_rank()}] Train Epoch: {epoch} "
                            f"[{(batch_idx+1) * len(sequences)}/{len(train_loader.sampler)} "
                            f"({100.0 * (batch_idx+1) / len(train_loader):.0f}%)] "
                            f"Loss: {loss.item():.6f}",
                            flush=True
                        )
                    else:
                        print(
                            f"Train Epoch: {epoch} "
                            f"[{(batch_idx+1) * len(sequences)}/{len(train_loader.dataset)} "
                            f"({100.0 * (batch_idx+1) / len(train_loader):.0f}%)] "
                            f"Loss: {loss.item():.6f}",
                            flush=True
                        )
            else:
                loss.backward()
                optimizer.step()
                if ddp:
                    print(
                        f"[Rank {dist.get_rank()}] Train Epoch: {epoch} "
                        f"[{(batch_idx+1) * len(sequences)}/{len(train_loader.sampler)} "
                        f"({100.0 * (batch_idx+1) / len(train_loader):.0f}%)] "
                        f"Loss: {loss.item():.6f}",
                        flush=True
                    )
                else:
                    print(
                        f"Train Epoch: {epoch} "
                        f"[{(batch_idx+1) * len(sequences)}/{len(train_loader.dataset)} "
                        f"({100.0 * (batch_idx+1) / len(train_loader):.0f}%)] "
                        f"Loss: {loss.item():.6f}",
                        flush=True
                    )
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    def test(self, model, test_loader, device):
        """
        Evaluate the model on the test dataset (non-DDP version).
        """
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

    def test_ddp(self, model, test_loader, device):
        """
        Evaluate the model using DistributedDataParallel (DDP).
        """
        model.eval()
        local_correct = 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                probabilities = torch.sigmoid(outputs.squeeze())
                predictions = (probabilities > 0.5).long()
                local_correct += predictions.eq(labels.squeeze()).sum().item()

        # Reduce correct counts across all ranks
        correct_tensor = torch.tensor(local_correct, dtype=torch.long, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        global_correct = correct_tensor.item()
        global_accuracy = 100.0 * global_correct / len(test_loader.dataset)
        if dist.get_rank() == 0:
            print(
                f"Test set: Accuracy: {global_correct}/{len(test_loader.dataset)} "
                f"({global_accuracy:.2f}%)\n"
            )
        return global_accuracy

    def evaluate(self, model, test_loader, device):
        """
        Evaluate the model and return the accuracy along with predictions and true labels.
        """
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                probs = torch.sigmoid(outputs.squeeze()).cpu().numpy().tolist()
                labels = labels.cpu().view(-1).numpy().tolist()
                predictions.extend(probs)
                true_labels.extend(labels)
        acc = BaselineCNN.assess_acc(predictions, true_labels)
        return acc, predictions, true_labels

    # -----------------------
    # Utility Methods (Static)
    # -----------------------

    @staticmethod
    def assess_acc(predictions, targets, thresh=0.5):
        """
        Compute accuracy given prediction probabilities and true targets.
        """
        y = np.asarray(targets)
        y_hat = np.asarray(predictions)
        y_hat = np.uint8(y_hat > thresh)
        correct = np.sum(y == y_hat)
        acc = correct / len(y)
        return acc

    @staticmethod
    def save_checkpoint(model, optimizer, epoch, average_loss, accuracy, path):
        """
        Save model and optimizer state.
        """
        # If model is wrapped in DDP, the actual parameters live in model.module
        model_state_dict = model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss,
            "accuracy": accuracy,
        }, path)

    @staticmethod
    def seed_everything(seed=42):
        """
        Set all random seeds for reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def load_data(dataset=None, sep=","):
        """
        Load data from a file path or DataFrame.
        """
        if dataset:
            if isinstance(dataset, str):
                if dataset.endswith((".csv", ".txt", ".tsv")):
                    D = pd.read_csv(dataset, sep=sep)
                elif dataset.endswith(".xlsx"):
                    D = pd.read_excel(dataset)
                elif dataset.endswith(".json"):
                    with open(dataset, "r") as f:
                        D = json.load(f)
                else:
                    print(f"Unrecognized format of {dataset}")
                    D = None
            elif isinstance(dataset, pd.DataFrame):
                D = dataset
            else:
                print("Dataset must be a path or a pandas DataFrame.")
                D = None
        else:
            print(f"Dataset is {dataset}.")
            D = None
        return D

    @staticmethod
    def encode_dna(seq):
        """
        One-hot encode a DNA sequence.
        """
        encoding = {
            "A": [1, 0, 0, 0, 0],
            "C": [0, 1, 0, 0, 0],
            "G": [0, 0, 1, 0, 0],
            "T": [0, 0, 0, 1, 0],
            "U": [0, 0, 0, 0, 1],
            "N": [1/4, 1/4, 1/4, 1/4, 1/4],
        }
        return [encoding.get(base, [0, 0, 0, 0, 0]) for base in seq]

    # -----------------------
    # Run the Training/Evaluation Pipeline
    # -----------------------

    def run(self, args):
        """
        End-to-end run method that sets up data, trains (or evaluates) the model, and saves checkpoints.
        The input `args` is assumed to be an argparse.Namespace with the needed attributes.
        """
        # Define project home and extract parameters from args
        PROJ_HOME = os.path.join(os.path.expanduser("~/projects/mirLM"))
        mRNA_max_length = args.mRNA_max_len
        miRNA_max_length = args.miRNA_max_len
        device = torch.device(args.device)
        num_epochs = args.num_epochs
        batch_size = args.batch_size
        dataset_name = args.dataset_name
        add_linker = args.add_linker
        train_dataset_path = args.train_dataset_path
        val_dataset_path = args.val_dataset_path
        test_dataset_path = args.test_dataset_path
        evaluate_flag = args.evaluate
        ddp_flag = args.ddp
        resume_ckpt = args.resume_ckpt

        accumulation_step = args.accumulation_step
        model_name = "CNN"

        # Initialize DDP if needed
        if ddp_flag:
            BaselineCNN.seed_everything(seed=42)
            local_rank = int(os.environ["LOCAL_RANK"])
            dist.init_process_group(backend="nccl", init_method="env://")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        else:
            local_rank = 0
            rank = 0
            world_size = 1

        if rank == 0:
            print("------ CNN Binary Classification starts here ------")
            print(f"mRNA length = {mRNA_max_length}")
            print(f"On device {device}")
            print(f"For {num_epochs} epochs")
            print(f"DDP = {ddp_flag}, World size = {world_size}, Rank = {rank}")

        # Load configuration (e.g., learning rate, weight decay, etc.)
        config_path = os.path.join(PROJ_HOME, "checkpoints", "CNN", "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        lr = config["learning_rate"] * world_size
        weight_decay = config["weight_decay"]

        # Create dataloaders
        if evaluate_flag:
            D_test = BaselineCNN.load_data(test_dataset_path)
            test_dataset = BaselineCNNDataset(
                dataset=D_test,
                mRNA_max_length=mRNA_max_length,
                miRNA_max_length=miRNA_max_length,
                add_linker = add_linker,
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            D_train = BaselineCNN.load_data(train_dataset_path)
            D_val = BaselineCNN.load_data(val_dataset_path)
            train_dataset = BaselineCNNDataset(
                dataset=D_train,
                mRNA_max_length=mRNA_max_length,
                miRNA_max_length=miRNA_max_length,
                mRNA_col="mRNA sequence",
                add_linker=add_linker,
            )
            val_dataset = BaselineCNNDataset(
                dataset=D_val,
                mRNA_max_length=mRNA_max_length,
                miRNA_max_length=miRNA_max_length,
                mRNA_col="mRNA sequence",
                add_linker = add_linker,
            )
            if ddp_flag:
                train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                shuffle_data = False
            else:
                train_sampler = None
                val_sampler = None
                shuffle_data = True
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle_data)
            test_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False)

        # Move model to device.
        self.to(device)

        # If in evaluation mode, load checkpoint and run evaluation.
        if evaluate_flag:
            ckpt_path = os.path.join(PROJ_HOME, "checkpoints", dataset_name, "CNN", str(mRNA_max_length), "checkpoint_epoch_final.pth")
            loaded_data = torch.load(ckpt_path, map_location=device)
            self.load_state_dict(loaded_data['model_state_dict'])
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            # If using DDP, wrap the model.
            model = self
            if ddp_flag:
                model = nn.parallel.DistributedDataParallel(
                    self,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False
                )
            loss_fn = nn.BCELoss()
            optimizer = optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            train_loss_list = []
            test_accuracy_list = []

        # Evaluation-only mode.
        if evaluate_flag:
            test_dataset_name = os.path.basename(test_dataset_path).split('.')[0]
            acc, predictions, true_labels = self.evaluate(self, test_loader, device)
            print("Evaluation accuracy =", acc)
            output_path = os.path.join(PROJ_HOME, "Performance", test_dataset_name, model_name)
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, f"predictions_{mRNA_max_length}.json"), "w") as f:
                json.dump({"predictions": predictions, "true_labels": true_labels}, f)
            print("Evaluation completed. Predictions saved to", output_path)
        else:
            start_epoch = 0
            if resume_ckpt is not None:
                if rank == 0:
                    print(f"Resuming from checkpoint: {resume_ckpt}")
                loaded_data = torch.load(resume_ckpt, map_location=device)
                start_epoch = loaded_data["epoch"] + 1  # resume from the next epoch
                # If using DDP, load into model.module
                if ddp_flag and isinstance(self, nn.parallel.DistributedDataParallel):
                    self.module.load_state_dict(loaded_data["model_state_dict"])
                else:
                    self.load_state_dict(loaded_data["model_state_dict"])
                optimizer.load_state_dict(loaded_data["optimizer_state_dict"])
                if ddp_flag:
                    dist.barrier()

            final_epoch = start_epoch + num_epochs
            model_checkpoint_dir = os.path.join(PROJ_HOME, "checkpoints", dataset_name, "CNN", str(mRNA_max_length))
            os.makedirs(model_checkpoint_dir, exist_ok=True)

            start_time = time.time()
            for epoch in range(start_epoch, final_epoch):
                if ddp_flag:
                    train_loader.sampler.set_epoch(epoch)
                # Use the (possibly DDP-wrapped) model for forward passes.
                train_loss = self.train_model(model, train_loader, loss_fn, optimizer, device, epoch, accumulation_step, ddp_flag)
                if ddp_flag:
                    accuracy = self.test_ddp(model, test_loader, device)
                else:
                    accuracy = self.test(model, test_loader, device)
                if rank == 0:
                    train_loss_list.append(train_loss)
                    test_accuracy_list.append(accuracy)
                    cost = time.time() - start_time
                    remain = (cost / (epoch + 1)) * (num_epochs - epoch - 1) / 3600  # remaining time in hours
                    print(f"Remaining: {remain:.2f} hours")
                if epoch % 10 == 0 and rank == 0:
                    BaselineCNN.save_checkpoint(model, optimizer, epoch, train_loss, accuracy,
                                                os.path.join(model_checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))

            if rank == 0:
                BaselineCNN.save_checkpoint(model, optimizer, epoch, train_loss, accuracy,
                                            os.path.join(model_checkpoint_dir, f"checkpoint_epoch_final.pth"))
                time_taken = time.time() - start_time
                print("Time taken for {} epochs = {} min.".format(num_epochs, time_taken / 60))
                perf_dir = os.path.join(PROJ_HOME, "Performance", dataset_name, model_name)
                os.makedirs(perf_dir, exist_ok=True)
                with open(os.path.join(perf_dir, f"evaluation_accuracy_{mRNA_max_length}.json"), "w") as fp:
                    json.dump(test_accuracy_list, fp)
                with open(os.path.join(perf_dir, f"train_loss_{mRNA_max_length}.json"), "w") as fp:
                    json.dump(train_loss_list, fp)

            if ddp_flag:
                dist.destroy_process_group()


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Run training script for binary classification.")
    parser.add_argument("--mRNA_max_len", type=int, default=1000,
                        help="Maximum length of mRNA sequences (default: 1000)")
    parser.add_argument("--miRNA_max_len", type=int, default=28,
                        help="Maximum length of miRNA sequences (default: 28)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run training on (default: auto-detected)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size to load dataset")
    parser.add_argument("--accumulation_step", type=int, required=False, default=2, help="Accumulation of batch size before updating model parameters")
    parser.add_argument("--dataset_name", type=str, default="training data",
                        help="Name of the folder to save training performance and checkpoints")
    parser.add_argument("--no_linker", action="store_false", dest="add_linker", required=False,
                        help="Disabling linker between mRNA and miRNA seqs.")
    parser.add_argument("--train_dataset_path", type=str, default="path/to/training/data",
                        help="Path to training data")
    parser.add_argument("--val_dataset_path", type=str, default="path/to/validation/data",
                        help="Path to validation data")
    parser.add_argument("--test_dataset_path", type=str, default="path/to/test/data",
                        help="Path to test data")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model on test dataset")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel for multi-GPU training")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    args = parser.parse_args()
    
    # Load config to determine model parameters (assumed to be in the same format as before)
    PROJ_HOME = os.path.join(os.path.expanduser("~/projects/mirLM"))
    config_path = os.path.join(PROJ_HOME, "checkpoints", "CNN", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Instantiate the model with an input size equal to (mRNA_max_len + miRNA_max_len)
    if args.add_linker:
        input_size = args.mRNA_max_len + 6 + args.miRNA_max_len # linker length = 6
    else:
        input_size = args.mRNA_max_len + args.miRNA_max_len
    model = BaselineCNN(input_size=input_size,
                        num_classes=config["num_classes"],
                        kernel_size=config["kernel_size"])
    # Run training/evaluation.
    model.run(args)

