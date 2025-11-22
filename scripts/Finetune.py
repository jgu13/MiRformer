# Finetune the DTEA model on TargetScan_dataset/Merged_primates_finetune.csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import os
import sys
import math
import wandb
import random
import numpy as np
from time import time
from itertools import chain

from utils import load_dataset
from ckpt_util import load_training_state
from Data_pipeline import SpanDataset, BatchStratifiedSampler, CharacterTokenizer
from DTEA_model import DTEA
from Global_parameters import PROJ_HOME, YOUR_WANDB_API_KEY

data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")

# def load_model_ckpt(model, ckpt_path):
#     # Load the pretrained checkpoint (contains wrapper with base model + pretraining heads)
#     pretrain_dict = torch.load(ckpt_path, map_location=model.device)
    
#     # Create a filtered dict with only base model weights, removing 'base.' prefix
#     model_dict = {key[5:]: value for key, value in pretrain_dict.items() 
#                   if key.startswith('base.')}
    
#     # Get the current model's state dict to see what keys exist
#     current_model_dict = model.state_dict()
    
#     # Only load encoder weights (skip predictor heads that have architecture changes)
#     filtered_dict = {}
#     for key, value in model_dict.items():
#         # Skip predictor heads that might have architecture changes
#         if 'predictor.binding' in key or 'predictor.cleavage' in key or 'predictor.qa_outputs' in key:
#             print(f"Skipping predictor head weight: {key}")
#             continue
            
#         if key in current_model_dict and value.shape == current_model_dict[key].shape:
#             filtered_dict[key] = value
#         else:
#             print(f"Skipping incompatible weight: {key} (shape mismatch or key not found)")
    
#     # Load the compatible weights, strict=False to ignore missing keys
#     model.load_state_dict(filtered_dict, strict=False)
#     print(f"Loaded {len(filtered_dict)} compatible weights out of {len(model_dict)} total weights")
#     return model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run(ckpt_path,
        train_path,
        valid_path,
        test_path,  # Currently unused - reserved for future test evaluation
        device,
        epochs,
        accumulation_step,
        batch_size,
        seed,
        lr,
        mrna_max_len,
        mirna_max_len,
        predict_span,
        predict_binding,
        use_longformer,
        ):
    seed_everything(seed)
    tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                model_max_length=mrna_max_len,
                                padding_side="right")    
    model = DTEA(mrna_max_len=mrna_max_len,
                                mirna_max_len=mirna_max_len,
                                device=device,
                                epochs=epochs,
                                embed_dim=1024,
                                num_heads=8,
                                num_layers=4,
                                ff_dim=4096,
                                batch_size=batch_size,
                                lr=lr,
                                seed=seed,
                                predict_span=predict_span,
                                predict_binding=predict_binding,
                                predict_cleavage=False,
                                use_longformer=use_longformer)
    # load dataset
    D_train  = load_dataset(train_path, sep=',')
    D_val    = load_dataset(valid_path, sep=',')
    ds_train = SpanDataset(data=D_train,
                                    mrna_max_len=mrna_max_len,
                                    mirna_max_len=mirna_max_len,
                                    tokenizer=tokenizer,
                                    seed_start_col="seed start",
                                    seed_end_col="seed end",)
    ds_val = QuestionAnswerDataset(data=D_val,
                                mrna_max_len=mrna_max_len,
                                mirna_max_len=mirna_max_len,
                                tokenizer=tokenizer, 
                                seed_start_col="seed start",
                                seed_end_col="seed end",)
    train_sampler = BatchStratifiedSampler(labels = [example["target"].item() for example in ds_train],
                                    batch_size = batch_size)
    train_loader = DataLoader(ds_train, 
                        batch_sampler=train_sampler,
                        shuffle=False)
    val_loader   = DataLoader(ds_val, 
                            batch_size=batch_size,
                            shuffle=False)
    loss_fn   = nn.CrossEntropyLoss()
    model.to(device)
    
    # def set_requires_grad(m, flag):
    #     for p in m.parameters():
    #         p.requires_grad = flag

    # # Freeze everything first
    # set_requires_grad(model, False)

    # # Train heads + last block + norms
    # train_modules = [
    #     model.predictor.binding_head,    # MIL binding
    #     model.predictor.qa_outputs,      # span start/end
    #     model.predictor.cross_norm,      # LayerNorm after cross-attn
    # ]
    # for m in train_modules:
    #     set_requires_grad(m, True)

    # # Unfreeze the last block of each encoder
    # for enc in [model.predictor.mrna_encoder, model.predictor.mirna_encoder]:
    #     # keep lower blocks frozen
    #     if hasattr(enc, "layers"):
    #         last_block = enc.layers[-1]
    #         set_requires_grad(last_block, True)

    # # Keep cross-attention un-frozen
    # set_requires_grad(model.predictor.cross_attn_layer, True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Debug: Print what's trainable
    print(f"Total trainable parameters: {len(trainable_params)}")
    print("Trainable modules:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")
    
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-2)

    total_steps   = math.ceil(len(train_loader) * epochs / accumulation_step)
    warmup_steps  = int(0.05 * total_steps)  # 5% warmup (3–5% is typical)
    eta_min       = 3e-5                     # final floor  

    warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    if ckpt_path is not None:
        resume_data = load_training_state(
                            ckpt_path=ckpt_path, 
                            model=model, 
                            optimizer=optimizer, 
                            scheduler=scheduler, 
                            map_location=device)

    start    = time()
    count    = 0
    patience = 10
    best_binding_acc = 0
    best_exact_match = 0
    best_f1_score    = 0
    best_composite_metric = 0
    model_checkpoints_dir = os.path.join(
        PROJ_HOME, 
        "checkpoints", 
        "TargetScan", 
        "TwoTowerTransformer", 
        "Longformer",
        str(mrna_max_len),
        "finetune",
    )
    os.makedirs(model_checkpoints_dir, exist_ok=True)

    # weights and bias initialization
    # uncomment to use wandb
    # wandb.login(key=YOUR_WANDB_API_KEY)
    # wandb.init(
    #     project="mirna-Span-Prediction",
    #     name=f"DTEA_len:{mrna_max_len}-epoch:{epochs}-finetune-mean-pooling", 
    #     config={
    #         "batch_size": batch_size * accumulation_step,
    #         "epochs": epochs,
    #         "learning rate": lr,
    #     },
    #     tags=["binding-span", "longformer", "8-heads-4-layer", "finetune"],
    #     save_code=False,
    #     job_type="train"
    # )

    for epoch in range(epochs):
        # TRAINING
        train_loss = model.train_loop(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            accumulation_step=accumulation_step,
            trainable_params=trainable_params,
        )

        # EVALUATION
        eval_loss, acc_binding, acc_start, acc_end, exact_match, f1, acc_cleavage = model.eval_loop(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        # uncomment to use wandb
        # try:
        #     wandb.log({
        #         "epoch": epoch,
        #         "train/loss": train_loss,
        #         "eval/loss": eval_loss,
        #         "eval/binding accuracy": acc_binding,
        #         "eval/start accuracy": acc_start,
        #         "eval/end accuracy": acc_end,
        #         "eval/exact match": exact_match,
        #         "eval/F1 score": f1,
        #     }, step=epoch)
        # except Exception as e:
        #     print(f"[W&B] log failed at epoch {epoch}: {e}")

        # CHECK FOR IMPROVEMENT
        if predict_binding and predict_span:
            composite = f1 + acc_binding
            improved = composite > best_composite_metric
        elif predict_binding:
            improved = acc_binding >= best_binding_acc
        else:  # predict_span only
            improved = exact_match >= best_exact_match

        if improved:
            # update bests & reset patience
            if predict_binding and predict_span:
                best_composite_metric = composite
            best_binding_acc      = acc_binding   if predict_binding else best_binding_acc
            best_f1_score         = f1            if predict_span    else best_f1_score
            best_exact_match      = exact_match   if predict_span    else best_exact_match
            count = 0

            # save checkpoint
            ckpt_name = (
                f"best_composite_{best_f1_score:.4f}_{best_binding_acc:.4f}_epoch{epoch}.pth"
                if (predict_binding and predict_span)
                else f"best_binding_acc_{best_binding_acc:.4f}_epoch{epoch}.pth"
                if predict_binding
                else f"best_exact_match_{best_exact_match:.4f}_epoch{epoch}.pth"
            )
            ckpt_path = os.path.join(model_checkpoints_dir, ckpt_name)

            try:
                torch.save(model.state_dict(), ckpt_path)
                print(f"[CKPT] saved to {ckpt_path}", flush=True)
            except Exception as e:
                print(f"[CKPT][ERROR] failed to save {ckpt_path}: {e}", file=sys.stderr, flush=True)
            
        else:
            count += 1
            if count >= patience:
                print("Max patience reached with no improvement. Early stopping.")
                break

        # ETA printout
        elapsed = time() - start
        remaining = elapsed / (epoch + 1) * (epochs - epoch - 1) / 3600
        print(f"Still remain: {remaining:.2f} hrs.")

if __name__ == "__main__":
    mrna_max_len = 520
    mirna_max_len = 24
    ckpt_path = os.path.join(PROJ_HOME, "checkpoints/TargetScan/TwoTowerTransformer/Longformer/520/Pretrain_DDP/overall_accuracy_0.5110_epoch2.pth")
    train_path = os.path.join(data_dir, "TargetScan_finetune_train.csv")
    valid_path = os.path.join(data_dir, "TargetScan_finetune_validation.csv")

    run(ckpt_path=None,
        train_path=train_path,
        valid_path=valid_path,
        test_path="",
        accumulation_step=8,
        mrna_max_len=mrna_max_len,
        mirna_max_len=mirna_max_len,
        device="cuda:1",
        epochs=10,
        batch_size=32,
        lr=3e-5,
        seed=10020,
        predict_span=True,
        predict_binding=True,
        use_longformer=True)
