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


def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    loss_fn,
    mRNA_max_length,
    miRNA_max_length,
    log_interval=10,
    accumulation_step=None,
    ddp=False
):
    """
    Training loop.
    """
    model.train()
    epoch_loss = 0.0
    for batch_idx, (seq, seq_mask, target) in enumerate(train_loader):
        seq, seq_mask, target = seq.to(device), seq_mask.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(
            device=device,
            input_ids=seq,
            input_mask=seq_mask,
            max_mRNA_length=mRNA_max_length,
            max_miRNA_length=miRNA_max_length
        )
        loss = loss_fn(output.squeeze().sigmoid(), target.squeeze())
        if accumulation_step is not None:
            loss = loss / accumulation_step
            loss.backward()
            if (batch_idx + 1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                if ddp:
                    print(
                            f"[Rank {dist.get_rank()}] "
                            f"Train Epoch: {epoch} "
                            f"[{batch_idx * len(seq)}/{len(train_loader.sampler)} "
                            f"({100.0 * batch_idx / len(train_loader):.0f}%)] "
                            f"Loss: {loss.item():.6f}\n"
                        )
                else:
                    print(
                            f"Train Epoch: {epoch} "
                            f"[{batch_idx * len(seq)}/{len(train_loader.dataset)} "
                            f"({100.0 * batch_idx / len(train_loader):.0f}%)] "
                            f"Loss: {loss.item():.6f}\n"
                        )
        else:
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                        f"[Rank {dist.get_rank()}] "
                        f"Train Epoch: {epoch} "
                        f"[{batch_idx * len(seq)}/{len(train_loader.dataset) // int(os.environ['WORLD_SIZE'])} "
                        f"({100.0 * batch_idx / len(train_loader):.0f}%)] "
                        f"Loss: {loss.item():.6f}\n"
                    )
        epoch_loss += loss.item()
    average_loss = epoch_loss / len(train_loader)
    return average_loss

def test(model, device, test_loader, mRNA_max_length, miRNA_max_length):
    """Test loop."""
    model.eval()
    correct = 0
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
            probabilities = torch.sigmoid(output.squeeze())
            predictions = (probabilities > 0.5).long()
            correct += predictions.eq(target.squeeze()).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(
            correct, len(test_loader.dataset), accuracy
        )
    )
    return accuracy

def test_ddp(model, device, test_loader, mRNA_max_length, miRNA_max_length):
    """Test loop with ddp."""
    model.eval()
    local_correct = 0
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
            probabilities = torch.sigmoid(output.squeeze())
            predictions = (probabilities > 0.5).long()
            local_correct += predictions.eq(target.squeeze()).sum().item()

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


def load_data(dataset=None, sep=","):
    """
    Load dataset from a file or DataFrame.
    """
    if dataset:
        if isinstance(dataset, str):
            if dataset.endswith((".csv", ".txt", ".tsv")):
                data = pd.read_csv(dataset, sep=sep)
            elif dataset.endswith(".xlsx"):
                data = pd.read_excel(dataset)
            elif dataset.endswith(".json"):
                with open(dataset, "r", encoding="utf-8") as f:
                    data = pd.read_json(f)
            else:
                raise ValueError(f"Unrecognized format of {dataset}")
        elif isinstance(dataset, pd.DataFrame):
            data = dataset
        else:
            raise TypeError("Dataset must be a path or a pandas DataFrame.")
    else:
        raise ValueError("Dataset cannot be None.")
    return data

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

def run_train(mRNA_max_len, 
              miRNA_max_len, 
              dataset=None,
              dataset_name='', 
              epochs=10, 
              device="cpu", 
              batch_size=16,
              ddp=False,
              rank=0,
              world_size=1,
              local_rank=0,
              resume_ckpt=None):
    """
    Main entry point for training.
    """
    # experiment settings:
    max_length = mRNA_max_len + miRNA_max_len + 2 # +2 to account for special tokens, like EOS
    use_padding = True
    rc_aug = False  # reverse complement augmentation
    add_eos = False  # add end of sentence token
    accumulation_step = 256 // batch_size  # effectively change batch size to 256
    if rank == 0:
        print("Batch size == ", batch_size)

    # model to be used for training
    pretrained_model_name = "hyenadna-small-32k-seqlen"  # use None if training from scratch

    # we need these for the decoder head, if using
    use_head = True
    n_classes = 1

    # provide a backbone configuration, if None, pretrained model config.json will be loaded
    # if `pretrained_model_name` is defined. Otherwise,
    backbone_cfg = None

    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in [
        "hyenadna-tiny-1k-seqlen",
        "hyenadna-small-32k-seqlen",
        "hyenadna-medium-160k-seqlen",
        "hyenadna-medium-450k-seqlen",
        "hyenadna-large-1m-seqlen",
    ]:
        path = f"{PROJ_HOME}/checkpoints"
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
    # from scratch
    else:
        try:
            model = HyenaDNAModel(
                **backbone_cfg, use_head=use_head, n_classes=n_classes
            )
        except TypeError as exc:
            raise TypeError("backbone_cfg must not be NoneType.") from exc

    learning_rate = backbone_cfg["lr"] * world_size # scale learning rate accordingly
    weight_decay = backbone_cfg["weight_decay"]

    if rank == 0:
        print("learning rate = ", learning_rate)
        print("weight decay = ", weight_decay)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "U", "N"],  # add RNA characters, N is uncertain
        model_max_length=max_length,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )

    Dataset = load_data(dataset, sep=",")

    ds_train, ds_test = train_test_split(
        Dataset, test_size=0.2, random_state=34, shuffle=True
    )
    ds_train = miRawDataset(
        ds_train,
        mRNA_max_length=mRNA_max_len,
        miRNA_max_length=miRNA_max_len,
        tokenizer=tokenizer,
        use_padding=use_padding,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )
    ds_test = miRawDataset(
        ds_test,
        mRNA_max_length=mRNA_max_len,
        miRNA_max_length=miRNA_max_len,
        tokenizer=tokenizer,
        use_padding=use_padding,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )

    # ------ Distributed Samplers ------
    if ddp:
        train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(ds_test, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_data = False  # Sampler does the shuffling for train
    else:
        train_sampler = None
        test_sampler = None
        shuffle_data = True
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle_data)
    test_loader = DataLoader(ds_test, batch_size=batch_size, sampler=test_sampler, shuffle=False) 

    # loss function
    loss_fn = nn.BCELoss()

    # create optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.to(device)
    
    # ---------- DDP Wrapping -----------
    if ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    start_epoch = 0
    if resume_ckpt is not None:
        if rank == 0:
            print(f"Resuming from checkpoint: {resume_ckpt}")
            # Only rank=0 loads from disk, then we broadcast to other ranks
            loaded_data = torch.load(resume_ckpt, map_location=device)
            start_epoch = loaded_data["epoch"] + 1  # e.g., if ckpt epoch=20, we start at 21
            # Because we wrapped with DDP, load into model.module if it's DDP
            if isinstance(model, nn.parallel.DistributedDataParallel):
                model.module.load_state_dict(loaded_data["model_state_dict"])
            else:
                model.load_state_dict(loaded_data["model_state_dict"])
            optimizer.load_state_dict(loaded_data["optimizer_state_dict"])
        # If DDP, broadcast the states from rank=0 to all ranks
        if ddp:
            dist.barrier() # synchronize all processes
            # broadcast the epoch to all processes
            start_epoch = torch.tensor(start_epoch, dtype=torch.int, device=device)
            dist.broadcast(start_epoch, src=0)
            start_epoch = start_epoch.item()

    # Now we can continue from `start_epoch` to `start_epoch + epochs`
    final_epoch = start_epoch + epochs
    
    average_loss_list = []
    accuracy_list = []
    
    start = time.time()

    for epoch in range(start_epoch, final_epoch):
        if ddp:
            # Important: set epoch for DistributedSampler to shuffle data consistently
            train_loader.sampler.set_epoch(epoch)
                
        average_loss = train(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_fn=loss_fn,
            mRNA_max_length=mRNA_max_len,
            miRNA_max_length=miRNA_max_len,
            accumulation_step=accumulation_step,
            ddp=ddp
        )
        if ddp:
            accuracy = test_ddp(
                    model=model,
                    device=device,
                    test_loader=test_loader,
                    mRNA_max_length=mRNA_max_len,
                    miRNA_max_length=miRNA_max_len
                )
        else:
            accuracy = test(
                model=model,
                    device=device,
                    test_loader=test_loader,
                    mRNA_max_length=mRNA_max_len,
                    miRNA_max_length=miRNA_max_len
                )
        
        if rank == 0: 
            average_loss_list.append(average_loss)
            accuracy_list.append(accuracy)
        
        # save checkpoints on cuda:0 only
        if epoch % 10 == 0 and rank == 0:
            model_checkpoints_dir = os.path.join(
                PROJ_HOME, "checkpoints", dataset_name, "HyenaDNA", str(mRNA_max_len)
            )
            os.makedirs(model_checkpoints_dir, exist_ok=True)
            # Save the model checkpoint
            checkpoint_path = os.path.join(model_checkpoints_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, average_loss, accuracy, checkpoint_path)
        
        if rank == 0:
            cost = time.time() - start
            remain = cost/(epoch + 1) * (final_epoch - epoch - 1) /60/60
            print(f'still remain: {remain} hrs.')
    # save the final model
    if rank == 0:
        final_ckpt_path = os.path.join(model_checkpoints_dir, "checkpoint_epoch_final.pth")
        save_checkpoint(model, optimizer, epoch, average_loss, accuracy, final_ckpt_path)

    return average_loss_list, accuracy_list


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
        help="Batch size loaded on each device"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="training data",
        help="Name of the folder to save training performance and checkpoints"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="path/to/training data",
        help="Path to training data"
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use DistributedDataParallel for multi-GPU training."
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
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    device = args.device
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    ddp_flag = args.ddp
    resume_ckpt = args.resume_ckpt
    
    # Other fixed parameters
    miRNA_max_length = 28
    model_name = "HyenaDNA"
    
    # 1) Initialize process group if DDP is used
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
        print("Binary Classification -- Start training --")
        print("mRNA length:", mRNA_max_length)
        print(f"For {num_epochs} epochs")
        print("Using device:", device)
        print(f"DDP = {ddp_flag}, World size = {world_size}, Rank = {rank}")
        print(f"Resume from checkpoint path {resume_ckpt}")
    # fact check
    print(
        f"[Rank={rank}, local_rank={local_rank}] "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')} "
        f"device={device}, torch.cuda.current_device()={torch.cuda.current_device()}"
    )
    
    PROJ_HOME = os.getcwd()  # Assuming current working directory as project home

    # Launch training
    start = time.time()
    train_average_loss, test_accuracy = run_train(
        mRNA_max_len=mRNA_max_length,
        miRNA_max_len=miRNA_max_length,
        epochs=num_epochs,
        dataset=dataset_path,
        dataset_name=dataset_name,
        device=device,
        batch_size=batch_size,
        ddp=ddp_flag,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        resume_ckpt=resume_ckpt
    )
    time_taken = time.time() - start
    
    if rank == 0:
        print(f"Time taken for {num_epochs} epochs = {(time_taken / 60):.2f} min.")


        # Save test_accuracy and train_average_loss
        perf_dir = os.path.join(PROJ_HOME, "Performance", dataset_name, model_name)
        os.makedirs(perf_dir, exist_ok=True)

        with open(
            os.path.join(perf_dir, f"test_accuracy_{mRNA_max_length}.json"),
            "w",
            encoding="utf-8"
        ) as fp:
            json.dump(test_accuracy, fp)

        with open(
            os.path.join(perf_dir, f"train_loss_{mRNA_max_length}.json"),
            "w",
            encoding="utf-8"
        ) as fp:
            json.dump(train_average_loss, fp)
    # destroy process group to clean up
    if ddp_flag:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
