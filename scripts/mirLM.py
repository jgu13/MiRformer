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

# Local imports
from Hyena_layer import HyenaDNAPreTrainedModel, HyenaDNAModel
from Data_pipeline import CharacterTokenizer, miRawDataset

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

class mirLM(nn.Module):
    def __init__(
        self,
        mRNA_max_length: int,
        miRNA_max_length: int,
        train_dataset_path: str=None,
        val_dataset_path: str=None,
        test_dataset_path:str=None,
        dataset_name: str="",
        base_model_name: str="",
        evaluate: bool=False,
        device: str='cuda',
        epochs:int=100,
        batch_size:int=64,
        ddp:bool=False,
        resume_ckpt:str=None,
        model_name: str=None,
        # exps params
        use_padding:bool = True,
        rc_aug:bool = False, # reverse complement augmentation
        add_eos:bool = False, # add end of sentence token
        use_head:bool = True,
        accumulation_step:int=1,
        n_classes:int = 1,
        backbone_cfg=None,
        basemodel_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.mRNA_max_len=mRNA_max_length
        self.miRNA_max_len=miRNA_max_length
        self.epochs=epochs
        self.dataset_name=dataset_name
        self.base_model_name=base_model_name
        self.model_name=model_name if model_name else base_model_name # if model_name is not provided then base_model_name will be used 
        self.train_dataset_path=train_dataset_path
        self.val_dataset_path=val_dataset_path
        self.test_dataset_path=test_dataset_path
        self.evaluate_flag=evaluate
        self.batch_size=batch_size
        self.ddp=ddp
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
        self.accumulation_step = accumulation_step  # effectively change batch size to 256
        if self.rank == 0:
            print("Batch size == ", self.batch_size)

        # model to be used for training
        # pretrained_model_name = "hyenadna-small-32k-seqlen"  # use None if training from scratch
        pretrained_model_name = None
        
        # we need these for the decoder head, if using
        self.use_head = use_head
        self.n_classes = n_classes
        self.backbone_cfg = backbone_cfg

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
                                        config=self.backbone_cfg,
                                        device=self.device,
                                        use_head=self.use_head,
                                        n_classes=self.n_classes,
                                    )
            pretrained_model_name_or_path = os.path.join(path, pretrained_model_name)
            if os.path.isdir(pretrained_model_name_or_path):
                if backbone_cfg is None:
                    with open(
                        os.path.join(pretrained_model_name_or_path, "config.json"),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        self.backbone_cfg = json.load(f)
                elif isinstance(backbone_cfg, str) and backbone_cfg.endswith(".json"):
                    with open(backbone_cfg, "r", encoding="utf-8") as f:
                        self.backbone_cfg = json.load(f)
                else:
                    assert isinstance(
                        backbone_cfg, dict
                    ), "self-defined backbone config must be a dictionary."
        # from scratch
        else:
            try:
                if isinstance(backbone_cfg, str) and backbone_cfg.endswith(".json"):
                    with open(backbone_cfg, "r", encoding="utf-8") as f:
                        self.backbone_cfg = json.load(f)
                else:
                    assert isinstance(
                        backbone_cfg, dict
                    ), "self-defined backbone config must be a dictionary."
                self.hyena = HyenaDNAModel(
                    **self.backbone_cfg, 
                    use_head=use_head, 
                    n_classes=n_classes,
                )
            except TypeError as exc:
                raise TypeError("backbone_cfg must not be NoneType.") from exc
        
        if basemodel_cfg:
            with open(basemodel_cfg, "r", encoding="utf-8") as f:
                self.basemodel_cfg = json.load(f)
            self.learning_rate = self.basemodel_cfg["lr"] * self.world_size # scale learning rate accordingly
            self.weight_decay = self.basemodel_cfg["weight_decay"]
            self.accumulation_step = self.basemodel_cfg["accumulation_step"]
            self.alpha = self.basemodel_cfg.get("alpha", None)
            self.margin = self.basemodel_cfg.get("margin", None)
            print(f"learning rate = {self.learning_rate}\n"
                  f"weight decay = {self.weight_decay}\n"
                  f"Alpha = {self.alpha}\n"
                  f"Margin = {self.margin}\n"
                  f"Accumulation step = {self.accumulation_step}\n")

    @classmethod
    def create_model(cls, **kwargs):
        """
        Factory method that returns an instance of the desired subclass
        based on the 'model_name' parameter.
        """
        base_model_name = kwargs.get("base_model_name", None)
        if base_model_name is None:
            raise ValueError("A 'base_model_name' must be provided!")
        
        if base_model_name == "TwoTowerMLP":
            # Import locally to avoid circular dependency
            from TwoTowerMLP import TwoTowerMLP
            return TwoTowerMLP(**kwargs)
        elif base_model_name == "HyenaDNA":
            from HyenaDNAWrapper import HyenaDNAWrapper
            return HyenaDNAWrapper(**kwargs)
        else:
            raise ValueError(f"Unknown base_model_name: {base_model_name}")
    
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
                        model,
                        optimizer, 
                        epoch, 
                        average_loss, 
                        accuracy, 
                        path,):
        """
        Helper function for saving model/optimizer state dicts.
        Only rank 0 should save to avoid file corruption.
        """
        model_state_dict = model.module.state_dict() \
                            if isinstance(model, nn.parallel.DistributedDataParallel) \
                            else model.state_dict()

        torch.save({
            "epoch": epoch,
            "model_state_dict": model_state_dict,
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
        
    def generate_perturbed_seeds(
                            self,
                            original_seq, 
                            original_mask,
                            seed_start, 
                            seed_end,
                            tokenizer):
        # print("Original sequence = ", original_seq)
        special_chars = ["[PAD]","[UNK]","[MASK]"]
        perturbed_seqs = []
        perturbed_masks = []
        original_seq = [tokenizer._convert_id_to_token(d.item()) for d in original_seq]
        seed_start = int(seed_start.item()) if torch.is_tensor(seed_start) else int(seed_start)
        seed_end = int(seed_end.item()) if torch.is_tensor(seed_end) else int(seed_end)
        if seed_start < len(original_seq) and seed_end < len(original_seq):
            seed_match = original_seq[seed_start:seed_end]
            # print("Seed start = ", seed_start)
            # print("Seed end = ", seed_end)
            # print("Seed match region = ", seed_match)
            for i in range(len(seed_match)):
                if seed_match[i] in special_chars:
                    continue
                for base in ["A", "T", "C", "G"]:
                    if base != seed_match[i]:
                        mutated = seed_match[:i] + [base] + seed_match[i+1:]
                        perturbed_seq = original_seq[:seed_start] + mutated + original_seq[seed_end:]
                        # convert back to ids
                        perturbed_seq = [tokenizer._convert_token_to_id(c) for c in perturbed_seq]
                        perturbed_seqs.append(torch.tensor(perturbed_seq, dtype=torch.long))
                        perturbed_masks.append(original_mask)
        else:
            print("Seed start >= length of sequence or seed end >= length of sequence. Skipping the current sequence perturbation.")
        return perturbed_seqs, perturbed_masks # 3 * seed_len
    
    def run(self, model):
        self.seed_everything(seed=42) 
           
        if self.rank == 0:
            print(
                "Binary Classification -- Start training --\n"
                f"mRNA length: {self.mRNA_max_len}\n"
                f"For {self.epochs} epochs\n"
                f"Using device: {self.device}\n"
                f"DDP = {self.ddp}, World size = {self.world_size}, Rank = {self.rank}\n"
                f"Resume from checkpoint path {self.resume_ckpt}\n"
            )
            # fact check
            # print(
            #     f"[Rank={rank}, local_rank={local_rank}] "
            #     f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')} "
            #     f"device={device}, torch.cuda.current_device()={torch.cuda.current_device()}"
            # )
        
        # wrap self with a local variable, call training, testing, evaluting on model instead of self
        
        if self.evaluate_flag:
            ckpt_path = os.path.join(PROJ_HOME, 
                                     "checkpoints", 
                                     self.dataset_name, 
                                     self.model_name, 
                                     str(self.mRNA_max_len), 
                                     "checkpoint_epoch_final.pth")
            loaded_data = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(loaded_data["model_state_dict"])
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
            if self.base_model_name == 'HyenaDNA':
                ds_test = miRawDataset(
                    D_test,
                    mRNA_max_length=self.mRNA_max_len, # pad to mRNA max length
                    miRNA_max_length=self.miRNA_max_len, # pad to miRNA max length
                    seed_start_col="seed start",
                    seed_end_col="seed end",
                    tokenizer=tokenizer,
                    use_padding=self.use_padding,
                    rc_aug=self.rc_aug,
                    add_eos=self.add_eos,
                    concat=True,
                    add_linker=True,
                )
            elif self.base_model_name == 'TwoTowerMLP':
                ds_test = miRawDataset(
                    D_test,
                    mRNA_max_length=self.mRNA_max_len, # pad to mRNA max length
                    miRNA_max_length=self.miRNA_max_len, # pad to miRNA max length
                    seed_start_col="seed start",
                    seed_end_col="seed end",
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
            
            if self.base_model_name == 'HyenaDNA':
                ds_train = miRawDataset(
                    D_train,
                    mRNA_max_length=self.mRNA_max_len,
                    miRNA_max_length=self.miRNA_max_len,
                    seed_start_col="seed start",
                    seed_end_col="seed end",
                    tokenizer=tokenizer,
                    use_padding=self.use_padding,
                    rc_aug=self.rc_aug,
                    add_eos=self.add_eos,
                    concat=True,
                    add_linker=True,
                )
                ds_val = miRawDataset(
                    D_val,
                    mRNA_max_length=self.mRNA_max_len,
                    miRNA_max_length=self.miRNA_max_len,
                    seed_start_col="seed start",
                    seed_end_col="seed end",
                    tokenizer=tokenizer,
                    use_padding=self.use_padding,
                    rc_aug=self.rc_aug,
                    add_eos=self.add_eos,
                    concat=True,
                    add_linker=True,
                )
            elif self.base_model_name == 'TwoTowerMLP':
                ds_train = miRawDataset(
                    D_train,
                    mRNA_max_length=self.mRNA_max_len,
                    miRNA_max_length=self.miRNA_max_len,
                    seed_start_col="seed start",
                    seed_end_col="seed end",
                    tokenizer=tokenizer,
                    use_padding=self.use_padding,
                    rc_aug=self.rc_aug,
                    add_eos=self.add_eos,
                    concat=False,
                )
                ds_val = miRawDataset(
                    D_val,
                    mRNA_max_length=self.mRNA_max_len,
                    miRNA_max_length=self.miRNA_max_len,
                    seed_start_col="seed start",
                    seed_end_col="seed end",
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
                val_sampler = DistributedSampler(ds_val, 
                                                  num_replicas=self.world_size, 
                                                  rank=self.rank, 
                                                  shuffle=False)
                shuffle_data = False  # Sampler does the shuffling for train
            else:
                train_sampler = None
                val_sampler = None
                shuffle_data = True
            
            train_loader = DataLoader(ds_train, 
                                      batch_size=self.batch_size, 
                                      sampler=train_sampler, 
                                      shuffle=shuffle_data)
            test_loader = DataLoader(ds_val, 
                                     batch_size=self.batch_size, 
                                     sampler=val_sampler, 
                                     shuffle=False) 
        
        if self.evaluate_flag:
            model.to(self.device)
            test_dataset_name = os.path.basename(self.test_dataset_path).split('.')[0]
            acc, predictions, true_labels = model.run_evaluation(
                model=model,
                test_loader = test_loader,
            )
            print("Evaluation Accuracy = ", acc)
            # Save predictions and true labels
            output_path = os.path.join(PROJ_HOME, "Performance", test_dataset_name, self.model_name)
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, f"predictions_{self.mRNA_max_len}.json"), "w") as f:
                json.dump({"predictions": predictions, "true_labels": true_labels}, f)

            print("Evaluation completed. Predictions saved to", output_path)
        else:
            # ---------- DDP Wrapping -----------
            if self.ddp:
                model = nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=False,
                    )
            
            # loss function
            loss_fn = nn.BCEWithLogitsLoss()

            # create optimizer
            optimizer = optim.AdamW(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

            # start training or evaluation
            model.to(self.device)
            
            start_epoch = 0
            
            if self.resume_ckpt is not None:
                if self.rank == 0:
                    print(f"Resuming from checkpoint: {self.resume_ckpt}")
                
                # Every process loads from disk, then we broadcast to other ranks
                loaded_data = torch.load(self.resume_ckpt, map_location=self.device)
                start_epoch = loaded_data["epoch"] + 1  # e.g., if ckpt epoch=20, we start at 21
                # Because we wrapped with DDP, load into model.module if it's DDP
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    model.module.load_state_dict(loaded_data["model_state_dict"])
                else:
                    model.load_state_dict(loaded_data["model_state_dict"])
                    
                optimizer.load_state_dict(loaded_data["optimizer_state_dict"])
                # If DDP, broadcast the states from rank=0 to all ranks
                if self.ddp:
                    dist.barrier() # synchronize all processes

            # Now we can continue from `start_epoch` to `start_epoch + epochs`
            final_epoch = start_epoch + self.epochs
            
            average_loss_list = []
            accuracy_list = []
            test_loss_list = []
            average_diff = []
            
            model_checkpoints_dir = os.path.join(
                        PROJ_HOME, 
                        "checkpoints", 
                        self.dataset_name, 
                        self.model_name, 
                        str(self.mRNA_max_len),
                    )
            os.makedirs(model_checkpoints_dir, exist_ok=True)
            
            start = time.time()
            best_acc = 0
            counter = 0 # counts epochs with no improvement
            patience = 10
            for epoch in range(start_epoch, final_epoch):
                if self.ddp:
                    # Important: set epoch for DistributedSampler to shuffle data consistently
                    train_loader.sampler.set_epoch(epoch)  
                         
                average_loss = self.run_training(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss_fn=loss_fn,
                    tokenizer=tokenizer,
                    margin=self.margin,
                    alpha=self.alpha,
                )
                accuracy, diff_score, test_loss = self.run_testing(
                    model=model,
                    test_loader=test_loader,
                    tokenizer=tokenizer,
                    loss_fn=loss_fn,
                    alpha=self.alpha,
                    margin=self.margin,
                )
                
                if self.rank == 0: 
                    average_loss_list.append(average_loss)
                    accuracy_list.append(accuracy)
                    average_diff.append(diff_score)
                    test_loss_list.append(test_loss)

                if accuracy >= best_acc and self.rank == 0:
                    best_acc = accuracy
                    counter = 0
                    # Save the model checkpoint
                    checkpoint_path = os.path.join(model_checkpoints_dir, 
                                                f"best_checkpoint_epoch_{epoch}.pth")
                    self.save_checkpoint(model=model,
                                        optimizer=optimizer, 
                                        epoch=epoch, 
                                        average_loss=average_loss, 
                                        accuracy=accuracy, 
                                        path=checkpoint_path)
                elif accuracy < best_acc:
                    counter += 1
                    # Check if early stopping condition is met.
                    if counter >= patience and self.rank == 0:
                        print(f"Early stopping triggered at epoch {epoch}. No improvement for {patience} consecutive epochs.")
                        final_epoch = epoch + 1
                        break

                cost = time.time() - start
                remain = cost/(epoch + 1) * (final_epoch - epoch - 1) /3600
                print(f'still remain: {remain} hrs.')

            if self.rank == 0:
                final_ckpt_path = os.path.join(model_checkpoints_dir, "checkpoint_epoch_final.pth")
                self.save_checkpoint(model=model, 
                                     optimizer=optimizer, 
                                     epoch=final_epoch - 1, 
                                     average_loss=average_loss, 
                                     accuracy=accuracy, 
                                     path=final_ckpt_path)
                time_taken = time.time() - start
                print(f"Time taken for {self.epochs} epochs = {(time_taken / 60):.2f} min.")
                
                # Save test_accuracy and train_average_loss
                perf_dir = os.path.join(PROJ_HOME, "Performance", self.dataset_name, self.model_name)
                os.makedirs(perf_dir, exist_ok=True)

                with open(
                    os.path.join(perf_dir, f"evaluation_accuracy_{self.mRNA_max_len}.json"),
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
                    
                with open(
                    os.path.join(perf_dir, f"evaluation_loss_{self.mRNA_max_len}.json"),
                    "w",
                    encoding="utf-8"
                ) as fp:
                    json.dump(test_loss_list, fp)
                    
        # destroy process group to clean up
        if self.ddp:
            dist.destroy_process_group()  