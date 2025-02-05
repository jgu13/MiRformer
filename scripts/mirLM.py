import os
import json
import time
import math
import torch
import random
import argparse
import numpy as np
from torch import nn, optim
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from Hyena_layer import HyenaDNAPreTrainedModel, HyenaDNAModel, LinearHead
from Data_pipeline import CharacterTokenizer, miRawDataset
from TwoTowerMLP import TwoTowerMLP

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

class mirLM(nn.Module):
    def __init__(
        self,
        mRNA_max_length: int,
        miRNA_max_length: int,
        train_dataset_path: str,
        val_dataset_path: str,
        test_dataset_path:str,
        dataset_name: str,
        model_name: str,
        evaluate_flag: bool,
        device: str,
        epochs:int,
        batch_size:int,
        ddp_flag:bool,
        resume_ckpt:str,
        # exps params
        use_padding:bool = True,
        rc_aug:bool = False, # reverse complement augmentation
        add_eos:bool = False, # add end of sentence token
        use_head:bool = True,
        n_classes:int = 1,
    ):
        self.mRNA_max_len=mRNA_max_length
        self.miRNA_max_len=miRNA_max_length
        self.epochs=epochs
        self.dataset_name=dataset_name
        self.model_name=model_name
        self.train_dataset_path=train_dataset_path
        self.val_dataset_path=val_dataset_path
        self.test_dataset_path=test_dataset_path
        self.evaluate_flag=evaluate_flag
        self.batch_size=batch_size
        self.ddp=ddp_flag
        self.resume_ckpt=resume_ckpt
        # DDP params
        # Initialize process group if DDP is used
        if self.ddp:
            dist.init_process_group(backend="nccl", init_method="env://")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = f"cuda:{self.local_rank}" if (torch.cuda.is_available() and self.ddp) else "cpu"
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.device = device

        # experiment settings:
        self.max_length = self.mRNA_max_len + self.miRNA_max_len + 2 # +2 to account for special tokens, like EOS
        self.use_padding = use_padding
        self.rc_aug = rc_aug  
        self.add_eos = add_eos  
        self.accumulation_step = 256 // self.batch_size  # effectively change batch size to 256
        if self.rank == 0:
            print("Batch size == ", self.batch_size)

        # model to be used for training
        pretrained_model_name = "hyenadna-small-32k-seqlen"  # use None if training from scratch

        # we need these for the decoder head, if using
        self.use_head = use_head
        self.n_classes = n_classes

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
            self.hyena = HyenaDNAPreTrainedModel.from_pretrained(
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
                if isinstance(backbone_cfg, str) and backbone_cfg.endswith(".json"):
                    with open(backbone_cfg, "r", encoding="utf-8") as f:
                        backbone_cfg = json.load(f)
                else:
                    assert isinstance(
                        backbone_cfg, dict
                    ), "self-defined backbone config must be a dictionary."
                self.hyena = HyenaDNAModel(
                    **backbone_cfg, 
                    use_head=use_head, 
                    n_classes=n_classes,
                )
            except TypeError as exc:
                raise TypeError("backbone_cfg must not be NoneType.") from exc
            
        if model_name == "HyenaDNA":
            self.model == self.hyena
        elif model_name == "TwoTowerMLP":
            self.model == TwoTowerMLP(
                        mRNA_max_length=mRNA_max_length,
                        miRNA_max_length=miRNA_max_length,
                        train_dataset_path=train_dataset_path,
                        val_dataset_path=val_dataset_path,
                        test_dataset_path=test_dataset_path,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        device=device,
                        epochs=epochs,
                        batch_size=batch_size,
                        ddp_flag=ddp_flag,
                        resume_ckpt=resume_ckpt,
                        evaluate_flag=evaluate_flag,
                        # exps params
                        use_padding=use_padding,
                        use_head=use_head,
                        rc_aug=rc_aug,
                        add_epos=add_eos,
                        )
        
        self.learning_rate = backbone_cfg["lr"] * self.world_size # scale learning rate accordingly
        self.weight_decay = backbone_cfg["weight_decay"]  

    def load_dataset(self, dataset=None, sep=","):
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

    def save_checkpoint(self,
                        optimizer, 
                        epoch, 
                        average_loss, 
                        accuracy, 
                        path,):
        """
        Helper function for saving model/optimizer state dicts.
        Only rank 0 should save to avoid file corruption.
        """
        if self.model_name == 'HyenaDNA':
            # If wrapped in DDP, actual parameters live in model.module
            model_state_dict = self.hyena.module.state_dict() \
                            if isinstance(self.hyena, nn.parallel.DistributedDataParallel) \
                            else self.hyena.state_dict()

            torch.save({
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss,
                "accuracy": accuracy,
            }, 
            path,
        )
        elif self.model_name == 'TwoTowerMLP':
            if self.ddp:
                # If wrapped in DDP, actual parameters live in model.module
                MLP_head_state_dict = self.MLP_head.module.state_dict()
                Q_layer_state_dict = self.Q_layer.module.state_dict()
                KV_layer_state_dict = self.KV_layer.module.state_dict()
            else:
                MLP_head_state_dict = self.MLP_head.state_dict()
                Q_layer_state_dict = self.Q_layer.state_dict()
                KV_layer_state_dict = self.KV_layer.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": MLP_head_state_dict,
                    "Q_layer_state_dict": Q_layer_state_dict,
                    "KV_layer_state_dict": KV_layer_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": average_loss,
                    "accuracy": accuracy,
                },
                path,
            )
                    
    @staticmethod
    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # for cudnn, if reproducibility is needed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train(
        self,
        train_loader,
        optimizer,
        epoch,
        loss_fn,
        log_interval=10,
    ):
        """
        Training loop.
        """
        self.hyena.train()
        epoch_loss = 0.0
        for batch_idx, (seq, seq_mask, target) in enumerate(train_loader):
            seq, seq_mask, target = (
                seq.to(self.device), 
                seq_mask.to(self.device), 
                target.to(self.device)
            )
            optimizer.zero_grad()
            output = self.hyena(
                device=self.device,
                input_ids=seq,
                input_mask=seq_mask,
                use_only_miRNA=True,
                max_mRNA_length=self.mRNA_max_length,
                max_miRNA_length=self.miRNA_max_length
            )
            loss = loss_fn(output.squeeze().sigmoid(), target.squeeze())
            if self.accumulation_step is not None:
                loss = loss / self.accumulation_step
                loss.backward()
                if (batch_idx + 1) % self.accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if self.ddp:
                        print(
                                f"[Rank {dist.get_rank()}] "
                                f"Train Epoch: {epoch} "
                                f"[{(batch_idx + 1) * len(seq)}/{len(train_loader.sampler)} "
                                f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                                f"Loss: {loss.item():.6f}\n",
                                flush=True
                            )
                    else:
                        print(
                                f"Train Epoch: {epoch} "
                                f"[{(batch_idx + 1) * len(seq)}/{len(train_loader.dataset)} "
                                f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                                f"Loss: {loss.item():.6f}\n",
                                flush=True
                            )
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (batch_idx + 1) % log_interval == 0:
                    if self.ddp:
                        print(
                                f"[Rank {dist.get_rank()}] "
                                f"Train Epoch: {epoch} "
                                f"[{(batch_idx + 1) * len(seq)}/{len(train_loader.dataset) // int(os.environ['WORLD_SIZE'])} "
                                f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                                f"Loss: {loss.item():.6f}\n"
                            )
                    else:
                        print(
                                f"Train Epoch: {epoch} "
                                f"[{(batch_idx + 1) * len(seq)}/{len(train_loader.dataset)} "
                                f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                                f"Loss: {loss.item():.6f}\n"
                            )                        
            epoch_loss += loss.item()
        average_loss = epoch_loss / len(train_loader)
        return average_loss 
    
    def test(
        self, 
        test_loader, 
    ):
        """Test loop."""
        self.hyena.eval()
        if self.ddp:
            local_correct = 0
        else:
            correct = 0
        with torch.no_grad():
            for seq, seq_mask, target in test_loader:
                seq, target, seq_mask = (
                    seq.to(self.device),
                    target.to(self.device),
                    seq_mask.to(self.device),
                )
                output = self.hyena(
                    device=self.device,
                    input_ids=seq,
                    input_mask=seq_mask,
                    use_only_miRNA=True,
                    max_mRNA_length=self.mRNA_max_length,
                    max_miRNA_length=self.miRNA_max_length
                )
                probabilities = torch.sigmoid(output.squeeze())
                predictions = (probabilities > 0.5).long()
                if self.ddp:
                    local_correct += predictions.eq(target.squeeze()).sum().item()
                else:
                    correct += predictions.eq(target.squeeze()).sum().item()
        
        if self.ddp:
            # convert to gpu tensor
            correct_tensor = torch.tensor(local_correct, dtype=torch.long, device=self.device)
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
        else:
            accuracy = 100.0 * correct / len(test_loader.dataset)
            print(
                "\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(
                    correct, len(test_loader.dataset), accuracy
                )
            )
            return accuracy

    def evaluate(
        self, 
        test_loader, 
        ):
        """Test loop with ddp."""
        self.hyena.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for seq, seq_mask, target in test_loader:
                seq, target, seq_mask = (
                    seq.to(self.device),
                    target.to(self.device),
                    seq_mask.to(self.device),
                )
                output = self.hyena(
                    device=self.device,
                    input_ids=seq,
                    input_mask=seq_mask,
                    use_only_miRNA=True,
                    max_mRNA_length=self.mRNA_max_length,
                    max_miRNA_length=self.miRNA_max_length,
                )
                probabilities = torch.sigmoid(output.squeeze()).cpu().numpy().tolist()
                targets = target.cpu().view(-1).numpy().tolist()
                predictions.extend(probabilities)
                true_labels.extend(targets)

        return predictions, true_labels
    
    def run(self):
        self.seed_everything(seed=42)    
        if self.rank == 0:
            print(
                "Binary Classification -- Start training --"
                f"mRNA length: {self.mRNA_max_len}"
                f"For {self.epochs} epochs"
                f"Using device: {self.device}"
                f"DDP = {self.ddp}, World size = {self.world_size}, Rank = {self.rank}"
                f"Resume from checkpoint path {self.resume_ckpt}"
            )
            # fact check
            # print(
            #     f"[Rank={rank}, local_rank={local_rank}] "
            #     f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')} "
            #     f"device={device}, torch.cuda.current_device()={torch.cuda.current_device()}"
            # )
            print("learning rate = ", self.learning_rate)
            print("weight decay = ", self.weight_decay)
        
        if self.evaluate_flag:
            ckpt_path = os.path.join(PROJ_HOME, 
                                     "checkpoints", 
                                     self.dataset_name, 
                                     self.model_name, 
                                     str(self.mRNA_max_len), 
                                     "checkpoint_epoch_final.pth")
            loaded_data = torch.load(ckpt_path)
            self.model.load_state_dict(loaded_data["model_state_dict"])
            print(f"Loaded checkpoint from {ckpt_path}")

        # create tokenizer
        tokenizer = CharacterTokenizer(
            characters=["A", "C", "G", "T", "N"],  # add RNA characters, N is uncertain
            model_max_length=self.max_length,
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side="left",  # since HyenaDNA is causal, we pad on the left
        )

        if self.evaluate_flag:
            D_test = self.load_dataset(self.test_dataset_path)
            if self.model_name == 'HyenaDNA':
                ds_test = miRawDataset(
                    D_test,
                    mRNA_max_length=self.mRNA_max_len, # pad to mRNA max length
                    miRNA_max_length=self.miRNA_max_len, # pad to miRNA max length
                    tokenizer=tokenizer,
                    use_padding=self.use_padding,
                    rc_aug=self.rc_aug,
                    add_eos=self.add_eos,
                    concat=True,
                )
            elif self.model_name == 'TwoTowerMLP':
                ds_test = miRawDataset(
                    D_test,
                    mRNA_max_length=self.mRNA_max_len, # pad to mRNA max length
                    miRNA_max_length=self.miRNA_max_len, # pad to miRNA max length
                    tokenizer=tokenizer,
                    use_padding=self.use_padding,
                    rc_aug=self.rc_aug,
                    add_eos=self.add_eos,
                    concat=False,
                )
            test_loader = DataLoader(ds_test, batch_size=self.batch_size, shuffle=False)
        else:
            D_train = self.load_dataset(self.train_dataset_path)
            D_val = self.load_dataset(self.val_dataset_path)
            
            if self.model_name == 'HyenaDNA':
                ds_train = miRawDataset(
                    D_train,
                    mRNA_max_length=self.mRNA_max_len,
                    miRNA_max_length=self.miRNA_max_len,
                    tokenizer=tokenizer,
                    use_padding=self.use_padding,
                    rc_aug=self.rc_aug,
                    add_eos=self.add_eos,
                    concat=True,
                )
                ds_test = miRawDataset(
                    D_val,
                    mRNA_max_length=self.mRNA_max_len,
                    miRNA_max_length=self.miRNA_max_len,
                    tokenizer=tokenizer,
                    use_padding=self.use_padding,
                    rc_aug=self.rc_aug,
                    add_eos=self.add_eos,
                    concat=True,
                )
            elif self.model_name == 'TwoTowerMLP':
                ds_train = miRawDataset(
                    D_train,
                    mRNA_max_length=self.mRNA_max_len,
                    miRNA_max_length=self.miRNA_max_len,
                    tokenizer=tokenizer,
                    use_padding=self.use_padding,
                    rc_aug=self.rc_aug,
                    add_eos=self.add_eos,
                    concat=False,
                )
                ds_test = miRawDataset(
                    D_val,
                    mRNA_max_length=self.mRNA_max_len,
                    miRNA_max_length=self.miRNA_max_len,
                    tokenizer=tokenizer,
                    use_padding=self.use_padding,
                    rc_aug=self.rc_aug,
                    add_eos=self.add_eos,
                    concat=False,
                )
            # ------ Distributed Samplers ------
            if self.ddp:
                train_sampler = DistributedSampler(ds_train, 
                                                   num_replicas=self.world_size, 
                                                   rank=self.rank, 
                                                   shuffle=True)
                test_sampler = DistributedSampler(ds_test, 
                                                  num_replicas=self.world_size, 
                                                  rank=self.rank, 
                                                  shuffle=False)
                shuffle_data = False  # Sampler does the shuffling for train
            else:
                train_sampler = None
                test_sampler = None
                shuffle_data = True
            
            train_loader = DataLoader(ds_train, 
                                      batch_size=self.batch_size, 
                                      sampler=train_sampler, 
                                      shuffle=shuffle_data)
            test_loader = DataLoader(ds_test, 
                                     batch_size=self.batch_size, 
                                     sampler=test_sampler, 
                                     shuffle=False) 

            # loss function
            loss_fn = nn.BCELoss()

            # create optimizer
            if self.model_name == 'HyenaDNA':
                optimizer = optim.AdamW(
                    self.hyena.parameters()#, lr=self.learning_rate, weight_decay=self.weight_decay
                )
            elif self.model_name == 'TwoTowerMLP':
                optimizer = optim.AdamW(
                    self.TwoTowerMLP
                )

        # start training or evaluation
        self.model.to(self.device)
        
        if self.evaluate_flag:
            test_dataset_name = os.path.basename(self.test_dataset_path).split('.')[0]
            predictions, true_labels = self.model.evaluate(
                model=self.model,
                test_loader = test_loader,
            )
            if self.model_name == 'HyenaDNA':
                predictions, true_labels = self.evaluate_hyenadna(
                    model=self.hyenadna_model,
                    device=self.device,
                    test_loader=test_loader,
                    mRNA_max_length=self.mRNA_max_len,
                    miRNA_max_length=self.miRNA_max_len
                )
            elif self.model_name == "TwoTowerMLP":
                predictions, true_labels = self.evaluate_two_tower_MLP(
                    HyenaDNA_feature_extractor=self.HyenaDNA_feature_extractor,
                    MLP_head=self.MLP_head,
                    Q_layer=self.Q_layer,
                    KV_layer=self.KV_layer,
                    device=self.device,
                    test_loader=test_loader
                ) 
            # Save predictions and true labels
            output_path = os.path.join(PROJ_HOME, "Performance", test_dataset_name, self.model_name)
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, f"predictions_{self.mRNA_max_len}.json"), "w") as f:
                json.dump({"predictions": predictions, "true_labels": true_labels}, f)

            print("Evaluation completed. Predictions saved to", output_path)
        else:
            # ---------- DDP Wrapping -----------
            if self.ddp:
                if self.model_name == 'HyenaDNA':
                    self.hyena = nn.parallel.DistributedDataParallel(
                        self.hyena,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=False
                    )
                elif self.model_name == 'TwoTowerMLP':
                    self.MLP_head = nn.parallel.DistributedDataParallel(
                        self.MLP_head,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=False
                    )
                    self.Q_layer = nn.parallel.DistributedDataParallel(
                        self.Q_layer,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=False
                    )

            start_epoch = 0
            
            if self.resume_ckpt is not None:
                if self.rank == 0:
                    print(f"Resuming from checkpoint: {self.resume_ckpt}")
                    # Only rank=0 loads from disk, then we broadcast to other ranks
                    loaded_data = torch.load(self.resume_ckpt, map_location=self.device)
                    start_epoch = loaded_data["epoch"] + 1  # e.g., if ckpt epoch=20, we start at 21
                    if self.model_name == 'HyenaDNA':
                        # Because we wrapped with DDP, load into model.module if it's DDP
                        if isinstance(self.hyena, nn.parallel.DistributedDataParallel):
                            self.hyena.module.load_state_dict(loaded_data["model_state_dict"])
                        else:
                            self.hyena.load_state_dict(loaded_data["model_state_dict"])
                    elif self.model_name == 'TwoTowerMLP':
                        if isinstance(self.MLP_head, nn.parallel.DistributedDataParallel):
                            self.MLP_head.module.load_state_dict(loaded_data["model_state_dict"])
                        else:
                            self.MLP_head.load_state_dict(loaded_data["model_state_dict"])
                        if isinstance(self.Q_layer, nn.parallel.DistributedDataParallel):
                            self.Q_layer.module.load_state_dict(loaded_data["Q_layer_state_dict"])
                        else:
                            self.Q_layer.load_state_dict(loaded_data["Q_layer_state_dict"])
                        if isinstance(self.KV_layer, nn.parallel.DistributedDataParallel):
                            self.KV_layer.module.load_state_dict(loaded_data["KV_layer_state_dict"])
                        else:
                            self.KV_layer.load_state_dict(loaded_data["KV_layer_state_dict"])
                    optimizer.load_state_dict(loaded_data["optimizer_state_dict"])
                # If DDP, broadcast the states from rank=0 to all ranks
                if self.ddp:
                    dist.barrier() # synchronize all processes
                    # broadcast the epoch to all processes
                    start_epoch = torch.tensor(start_epoch, dtype=torch.int, device=self.device)
                    dist.broadcast(start_epoch, src=0)
                    start_epoch = start_epoch.item()

            # Now we can continue from `start_epoch` to `start_epoch + epochs`
            final_epoch = start_epoch + self.epochs
            
            average_loss_list = []
            accuracy_list = []
            
            model_checkpoints_dir = os.path.join(
                        PROJ_HOME, 
                        "checkpoints", 
                        self.dataset_name, 
                        self.model_name, 
                        str(self.mRNA_max_len)
                    )
            os.makedirs(model_checkpoints_dir, exist_ok=True)
            
            start = time.time()

            for epoch in range(start_epoch, final_epoch):
                if self.ddp:
                    # Important: set epoch for DistributedSampler to shuffle data consistently
                    train_loader.sampler.set_epoch(epoch)
                if self.model_name == 'HyenaDNA':        
                    average_loss = self.train(
                        train_loader=train_loader,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss_fn=loss_fn,
                    )
                elif self.model_name == 'TwoTowerMLP':
                    average_loss = self.train_two_tower_MLP(
                        train_loader=train_loader,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss_fn=loss_fn,
                        accumulation_step=self.accumulation_step,
                    )
                if self.model_name == 'HyenaDNA':
                    accuracy = self.test_hyenadna(
                            test_loader=test_loader,
                        )
                elif self.model_name == 'TwoTowerMLP':
                    accuracy = self.test_two_tower_MLP(
                        test_loader=test_loader
                    )
                
                if self.rank == 0: 
                    average_loss_list.append(average_loss)
                    accuracy_list.append(accuracy)
                
                # save checkpoints on cuda:0 only
                if (epoch + 1) % 10 == 0 and self.rank == 0:
                    # Save the model checkpoint
                    checkpoint_path = os.path.join(model_checkpoints_dir, 
                                                   f"checkpoint_epoch_{epoch}.pth")
                    
                    self.save_checkpoint(optimizer, 
                                         epoch, 
                                         average_loss, 
                                         accuracy, 
                                         checkpoint_path)
                
                if self.rank == 0:
                    cost = time.time() - start
                    remain = cost/(epoch + 1) * (final_epoch - epoch - 1) /60/60
                    print(f'still remain: {remain} hrs.')
            # save the final model
            if self.rank == 0:
                final_ckpt_path = os.path.join(model_checkpoints_dir, "checkpoint_epoch_final.pth")
                self.save_checkpoint(optimizer, epoch, average_loss, accuracy, final_ckpt_path)
                time_taken = time.time() - start
                print(f"Time taken for {self.epochs} epochs = {(time_taken / 60):.2f} min.")
                
                # Save test_accuracy and train_average_loss
                perf_dir = os.path.join(PROJ_HOME, "Performance", self.dataset_name, self.model_name)
                os.makedirs(perf_dir, exist_ok=True)

                with open(
                    os.path.join(perf_dir, f"test_accuracy_{self.mRNA_max_len}.json"),
                    "w",
                    encoding="utf-8"
                ) as fp:
                    json.dump(accuracy_list, fp)

                with open(
                    os.path.join(perf_dir, f"train_loss_{self.mRNA_max_len}.json"),
                    "w",
                    encoding="utf-8"
                ) as fp:
                    json.dump(average_loss_list, fp)
        # destroy process group to clean up
        if self.ddp:
            dist.destroy_process_group()  