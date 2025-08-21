import token
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import os
import math
import wandb
import random
import numpy as np
from time import time

from transformer_model import QuestionAnsweringModel
from multilevel_masking_pretraining import PairedPretrainWrapper, loss_stage1_baseline, loss_stage2_seed_mrna, loss_stage3_bispan
from Attention_regularization import kl_diag_seed_loss
from Data_pipeline import CharacterTokenizer, QuestionAnswerDataset
from utils import load_dataset

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
# PROJ_HOME = os.path.expanduser("/Users/jiayaogu/Documents/Li Lab/mirLM---Micro-RNA-generation-with-mRNA-prompt/")

def cosine_decay(total, step, min_factor=0.1):
    """
    returns lambda at the current step
    """
    # cosine to min_factor
    f = min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * step))
    return f

def pretrain_loop(
            model: torch.nn.Module, 
            pretrain_wrapper: torch.nn.Module,
            dataloader: DataLoader, 
            optimizer: torch.optim, 
            scheduler: torch.optim.lr_scheduler, 
            device: str, 
            epoch: int, 
            total_updates: int, 
            updates_per_epoch: int,
            mask_id: int, 
            pad_id: int, 
            vocab_size: int,
            accumulation_step=1, 
            sigma=1.0,):
    model.train()
    total_loss = 0.0
    loss_list = []
    pretrain_wrapper.train()
    for batch_idx, batch in enumerate(dataloader):
        for k in batch: batch[k] = batch[k].to(device)

        # === MLM forward ===
        logits_mr, logits_mi, attn_weights = pretrain_wrapper.forward_pair(
            mrna_ids=batch["mrna_input_ids"],
            mrna_mask=batch["mrna_attention_mask"],
            mirna_ids=batch["mirna_input_ids"],
            mirna_mask=batch["mirna_attention_mask"],
        )

        # === MLM loss (both) ===
        loss1 = loss_stage1_baseline(wrapper=pretrain_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size)  
        loss2 = loss_stage2_seed_mrna(wrapper=pretrain_wrapper, batch=batch, mask_id=mask_id, vocab_size=vocab_size)
        loss3 = loss_stage3_bispan(wrapper=pretrain_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size)
        loss_mlm = loss1 + loss2 + loss3

        # === KL regularization on seed rows (positives only) ===
        loss_reg = kl_diag_seed_loss(
            attn=attn_weights,
            seed_q_start=batch["start_positions"],
            seed_q_end=batch["end_positions"],
            q_mask=batch["mrna_attention_mask"],
            k_mask=batch["mirna_attention_mask"],
            y_pos=batch["target"],
            sigma=sigma,
            k_seed_start=1)
        # compute global update index (constant over an accumulation group)
        update_in_epoch = batch_idx // accumulation_step
        global_update = epoch * updates_per_epoch + update_in_epoch
        lamb = cosine_decay(total_updates, global_update, min_factor=0.1)
        loss = loss_mlm + lamb * loss_reg

        loss = loss / accumulation_step
        loss.backward()
        bs = batch['mrna_input_ids'].size(0)
        if accumulation_step != 1:
            loss_list.append(loss.item())
            if (batch_idx + 1) % accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None: scheduler.step()
                optimizer.zero_grad()
                print(
                    f"Train Epoch: {epoch} "
                    f"[{(batch_idx + 1) * bs}/{len(dataloader.dataset)} "
                    f"({(batch_idx + 1) * bs / len(dataloader.dataset) * 100:.0f}%)] "
                    f"Avg loss: {sum(loss_list) / len(loss_list):.6f}\n",
                    flush=True
                )
                loss_list = []
        total_loss += loss.item() * accumulation_step
    # After the loop, if gradients remain (for non-divisible number of batches)
    if (batch_idx + 1) % accumulation_step != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def evaluate_mlm(model, 
                pretrain_wrapper,
                dataloader, 
                device, 
                vocab_size,
                pad_id,
                mask_id):
    model.eval()
    pretrain_wrapper.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}         
        
        loss1, correct1, mr_y1, mi_y1 = loss_stage1_baseline(pretrain_wrapper, batch, pad_id, mask_id, evaluate=True)
        loss2, correct2, mr_y2 = loss_stage2_seed_mrna(pretrain_wrapper, batch, mask_id, vocab_size, evaluate=True)
        loss3, correct3, mr_y3, mi_y3 = loss_stage3_bispan(pretrain_wrapper, batch, pad_id, mask_id, vocab_size, evaluate=True)
        
        total_loss += (loss1 + loss2 + loss3).item()
        total_correct += correct1 + correct2 + correct3
        
        total_masked += sum(
            t.ne(-100).sum().item() for t in [mr_y1, mi_y1, mr_y2, mr_y3, mi_y3]
        )

    avg_loss = total_loss / len(dataloader)
    acc = total_correct / total_masked
    print(f"Evaluation MLM accuracy: {acc:.4f}")
    return avg_loss, acc

def run(epochs, 
        device, 
        accumulation_step=1, 
        mrna_max_len=520, 
        mirna_max_len=24, 
        embed_dim=256, 
        num_heads=8, 
        num_layers=4, 
        ff_dim=1024, 
        dropout_rate=0.1, 
        batch_size=32,
        lr=1e-4):
    tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                model_max_length=mrna_max_len,
                                padding_side="right")
    
    train_path = os.path.join(PROJ_HOME, "TargetScan_dataset/Positive_primates_train_500_randomized_start.csv")
    valid_path = os.path.join(PROJ_HOME, "TargetScan_dataset/Positive_primates_validation_500_randomized_start.csv")

    print(f"Loading training data from: {train_path}")
    ds_train = QuestionAnswerDataset(data=load_dataset(train_path, sep=','),
                                    mrna_max_len=mrna_max_len,
                                    mirna_max_len=mirna_max_len,
                                    tokenizer=tokenizer,
                                    seed_start_col="seed start",
                                    seed_end_col="seed end",)
    
    print(f"Loading validation data from: {valid_path}")
    ds_val = QuestionAnswerDataset(data=load_dataset(valid_path, sep=','),
                                  mrna_max_len=mrna_max_len,
                                  mirna_max_len=mirna_max_len,
                                  tokenizer=tokenizer,
                                  seed_start_col="seed start",
                                  seed_end_col="seed end",)
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    
    print("Creating model...")
    model = QuestionAnsweringModel(mrna_max_len=mrna_max_len,
                                   mirna_max_len=mirna_max_len,
                                   device=device,
                                   epochs=epochs,
                                   embed_dim=embed_dim,
                                   num_heads=num_heads,
                                   num_layers=num_layers,
                                   ff_dim=ff_dim,
                                   batch_size=batch_size,
                                   lr=3e-5,
                                   seed=10020,
                                   predict_span=True,
                                   predict_binding=True,
                                   use_longformer=True)
    
    print("Creating pretrain wrapper...")
    pretrain_wrapper = PairedPretrainWrapper(
                        base_model = model, 
                        vocab_size = tokenizer.vocab_size, 
                        d_model    = embed_dim,
                        embed_weight = model.predictor.sn_embedding.weight)

    optimizer = AdamW(pretrain_wrapper.parameters(), lr=lr)

    wandb.login(key="600e5cca820a9fbb7580d052801b3acfd5c92da2")
    run = wandb.init(
        project="mirna-pretraining",
        name=f"Pretrain:{mrna_max_len}-epoch:{epochs}", 
        config={
            "batch_size": batch_size * accumulation_step,
            "epochs": epochs,
            "learning_rate": lr,
        },
        tags=["Pre-train", "MLM", "Attn_reg"],
        save_code=True,
        job_type="train"
    )

    model.to(device)
    pretrain_wrapper.to(device)
    
    print("Setting up training...")
    start = time()
    patience = 10
    best_accuracy = 0
    count = 0  # Initialize count variable for early stopping
    model_checkpoints_dir = os.path.join(
        PROJ_HOME, 
        "checkpoints", 
        "TargetScan", 
        "Pretrain", 
        str(mrna_max_len),
    )
    os.makedirs(model_checkpoints_dir, exist_ok=True)
    print(f"   Checkpoint directory: {model_checkpoints_dir}")
    
    steps_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / accumulation_step)
    total_updates = epochs * updates_per_epoch
    warmup_updates = int(0.05 * total_updates)
    eta_min = 3e-5

    warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_updates)
    cosine = CosineAnnealingLR(optimizer, T_max=total_updates - warmup_updates, eta_min=eta_min)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_updates])

    print("Starting training loop...")
    for epoch in range(epochs):
        print(f"   Starting epoch {epoch+1}/{epochs}")
        train_loss = pretrain_loop(
            model=model,
            pretrain_wrapper=pretrain_wrapper,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            total_updates=total_updates, 
            updates_per_epoch=updates_per_epoch,
            mask_id=tokenizer.mask_token_id,
            pad_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
            accumulation_step=accumulation_step,
            sigma=1.0,  
            )
        print(f"   Epoch {epoch+1} training completed, loss: {train_loss:.6f}")
        
        print(f"   Starting evaluation for epoch {epoch+1}...")
        eval_loss, acc = evaluate_mlm(
            model=model,
            pretrain_wrapper=pretrain_wrapper,
            dataloader=val_loader,
            device=device,
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_token_id,
            mask_id=tokenizer.mask_token_id,
        )
        wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "eval/loss": eval_loss,
                    "eval/token accuracy": acc
                }, step=epoch)
        
        if acc > best_accuracy:
            best_accuracy = acc
            ckpt_name = f"best_accuracy_{best_accuracy:.4f}_epoch{epoch}.pth"
            ckpt_path = os.path.join(model_checkpoints_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)

            model_art = wandb.Artifact(
                name="Pretrained-model",
                type="model",
                metadata={
                    "epoch": epoch,
                    "accuracy": best_accuracy
                }
            )
            model_art.add_file(ckpt_path)
            try:
                run.log_artifact(model_art, aliases=["best-pretrain"])
            except Exception as e:
                print(f"[W&B] artifact log failed at epoch {epoch}: {e}")
        else:
            count += 1
            if count >= patience:
                print("Max patience reached with no improvement. Early stopping.")
                break
        
        elapsed = time() - start
        remaining = elapsed / (epoch + 1) * (epochs - epoch - 1) / 3600
        print(f"Still remain: {remaining:.2f} hrs.")

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda:0"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")
    run(epochs=1, 
        device=device, 
        embed_dim=1024,  
        ff_dim=2048,     
        batch_size=32, 
        accumulation_step=16)  