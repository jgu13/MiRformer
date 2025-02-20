import os
import torch
import random
import time
import numpy as np
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
# local import
from mirLM import mirLM
from argument_parser import get_argument_parser
from Data_pipeline import CharacterTokenizer, miRawDataset

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")

def generate_perturbed_seeds(original_seq, 
                            original_mask,
                            seed_start, 
                            seed_end,
                            tokenizer):
    special_chars = ["[PAD]","[UNK]","[MASK]"]
    perturbed_seqs = []
    perturbed_masks = []
    original_seq = [tokenizer._convert_id_to_token(d.item()) for d in original_seq]
    seed = original_seq[seed_start:seed_end]
    for i in range(len(seed)):
        if seed[i] in special_chars:
            pass
        for base in ["A", "T", "C", "G"]:
            if base != seed[i]:
                mutated = seed[:i] + [base] + seed[i+1:]
                perturbed_seq = original_seq[:seed_start] + mutated + original_seq[seed_end:]
                # convert back to ids
                perturbed_seq = [tokenizer._convert_token_to_id(c) for c in perturbed_seq]
                perturbed_seqs.append(torch.tensor(perturbed_seq, dtype=torch.long))
                perturbed_masks.append(original_mask)
    return perturbed_seqs, perturbed_masks # 3 * seed_len

def forward(
    model, 
    original_seqs,
    original_masks,
    seed_start,
    seed_end,
    tokenizer,
    ):
    # Original scores (s_w)
    s_w = model(original_seqs,
                original_masks) # (batchsize, 1)

    # Generate perturbed sequences for the seed region
    perturbed_seqs = []
    perturbed_masks = []
    repeats = [] # the number of time to repeat s_w
    for seq, mask, start, end in zip(original_seqs, original_masks, seed_start, seed_end):
        perturbed, perturbed_mask = generate_perturbed_seeds(original_seq=seq, 
                                                             original_mask=mask,
                                                             seed_start=start, 
                                                             seed_end=end,
                                                             tokenizer=tokenizer)
        perturbed_seqs.extend(perturbed)
        perturbed_masks.extend(perturbed_mask)
        repeats.append(len(perturbed))
    # print("repeats = ", repeats)
    repeats = torch.tensor(repeats, dtype=torch.long, device=model.device)
    # perturbed_seqs has shape (batchsize * seed_len * 3)
    # Compute perturbed scores (s_i)
    s_i = model(torch.stack(perturbed_seqs).to(model.device),
                torch.stack(perturbed_masks).to(model.device)) # (batchsize * seed_len * 3, 1)
    
    return s_w, s_i, repeats

def train(
    model,
    train_loader,
    optimizer,
    epoch,
    tokenizer,
    accumulation_step=1,
    log_interval=10,
    margin=0.1,
    alpha=0.1,
):
    model.train()
    loss_l = []
    epoch_loss = 0.0
    for i, (original_seqs, original_seqs_mask, seed_start, seed_end) in enumerate(train_loader):
        original_seqs, original_seqs_mask, seed_start, seed_end = (
            original_seqs.to(model.device),
            original_seqs_mask.to(model.device),
            seed_start.to(model.device),
            seed_end.to(model.device),
        )
        optimizer.zero_grad()
        s_w, s_i, repeats = forward(
                    model, 
                    original_seqs=original_seqs,
                    original_masks=original_seqs_mask,
                    seed_start=seed_start,
                    seed_end=seed_end,
                    tokenizer=tokenizer)
        s_w, s_i = (s_w.squeeze().sigmoid(), s_i.squeeze().sigmoid())
        s_w = torch.repeat_interleave(s_w, repeats) # (batchsize * seed_len * 3,)
        ranking_loss = 0.0
        # compute ranking loss
        ranking_loss = torch.clamp(s_i - s_w + margin, min=0.0).mean()
        if accumulation_step != 1:
            ranking_loss = ranking_loss / accumulation_step
            ranking_loss.backward()
            loss_l.append(ranking_loss.item())
            if (i + 1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(
                    f"Train Epoch: {epoch} "
                    f"[{(i + 1) * len(original_seqs)}/{len(train_loader.dataset)} "
                    f"({100.0 * (i + 1) / len(train_loader):.0f}%)] "
                    f"Avg Loss: {sum(loss_l) / len(loss_l):.6f}\n",
                    flush=True
                )
                loss_l = []
        else:
            ranking_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(
                f"Train Epoch: {epoch} "
                f"[{(i + 1) * len(original_seqs)}/{len(train_loader.dataset)} "
                f"({100.0 * (i + 1) / len(train_loader):.0f}%)] "
                f"Loss: {ranking_loss.item():.6f}\n",
                flush=True
            )
        epoch_loss += ranking_loss.item()
    average_loss = epoch_loss / len(train_loader)
    return average_loss
        
class perturbDataset(Dataset):
    def __init__(
        self, 
        data,
        mRNA_max_length=40,
        miRNA_max_length=26, 
        tokenizer=None,
        use_padding=None,
        rc_aug=None,
        add_eos=False,
        concat=False,
        add_linker=False,
    ):
        self.use_padding = use_padding
        self.mRNA_max_length = mRNA_max_length
        self.miRNA_max_length = miRNA_max_length
        self.tokenizer = tokenizer
        self.rc_aug = rc_aug
        self.add_eos = add_eos
        self.concat = concat
        self.data = data
        self.add_linker = add_linker
        # Assuming the last column is the label, adjust this if needed
        self.mRNA_sequences = self.data[["mRNA sequence"]].values
        self.miRNA_sequences = self.data[["miRNA sequence"]].values
        self.seed_starts = self.data[["seed start"]].values
        self.seed_ends = self.data[["seed end"]].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        mRNA_seq = self.mRNA_sequences[idx][0]
        miRNA_seq = self.miRNA_sequences[idx][0]
        seed_start = self.seed_starts[idx][0]
        seed_end = self.seed_ends[idx][0]
        
        miRNA_seq = miRNA_seq.replace('U', 'T')[::-1]
        
        mRNA_seq_encoding = self.tokenizer(
            mRNA_seq,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.mRNA_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        mRNA_seq_tokens = mRNA_seq_encoding["input_ids"]  # get input_ids
        mRNA_seq_mask = mRNA_seq_encoding["attention_mask"]  # get attention mask
        # print("Tokenized sequence length = ", len(mRNA_seq_tokens))

        miRNA_seq_encoding = self.tokenizer(
            miRNA_seq,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.miRNA_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        miRNA_seq_tokens = miRNA_seq_encoding["input_ids"]  # get input_ids
        miRNA_seq_mask = miRNA_seq_encoding["attention_mask"]  # get attention mask
        
        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            mRNA_seq_tokens.append(self.tokenizer.sep_token_id)
            miRNA_seq_tokens.append(self.tokenizer.sep_token_id)
            mRNA_seq_mask.append(1)  # do not mask eos token
            miRNA_seq_mask.append(1)  # do not mask eos token
        
        if self.concat:
            if self.add_linker:
                linker = [self.tokenizer._convert_token_to_id("N")] * 6
                concat_seq_tokens = mRNA_seq_tokens + linker + miRNA_seq_tokens
                linker_mask = [1] * 6
                concat_seq_mask = mRNA_seq_mask + linker_mask + miRNA_seq_mask
            else:
                # concatenate miRNA and mRNA tokens
                concat_seq_tokens = mRNA_seq_tokens + miRNA_seq_tokens
                concat_seq_mask = mRNA_seq_mask + miRNA_seq_mask
            # convert to tensor
            concat_seq_tokens = torch.tensor(concat_seq_tokens, dtype=torch.long)
            concat_seq_mask = torch.tensor(concat_seq_mask, dtype=torch.long)
            return concat_seq_tokens, concat_seq_mask, seed_start, seed_end
        else:
            mRNA_seq_tokens = torch.tensor(mRNA_seq_tokens, dtype=torch.long)
            miRNA_seq_tokens = torch.tensor(miRNA_seq_tokens, dtype=torch.long)
            mRNA_seq_mask = torch.tensor(mRNA_seq_mask, dtype=torch.long)
            miRNA_seq_mask = torch.tensor(miRNA_seq_mask, dtype=torch.long)
            return mRNA_seq_tokens, miRNA_seq_tokens, mRNA_seq_mask, miRNA_seq_mask, seed_start, seed_end
    
def load_model(**args_dict):
    # load model checkpoint
    model = mirLM.create_model(**args_dict)
    ckpt_path = os.path.join(PROJ_HOME, 
                            "checkpoints", 
                            "TargetScan", 
                            model.model_name, 
                            str(model.mRNA_max_len), 
                            "checkpoint_epoch_final.pth")
    loaded_data = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(loaded_data["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")
    return model

def viz_sequence(seq, changes, file_name=None):
    # Create a matrix for the sequence logo
    logo_matrix = lm.alignment_to_matrix([seq])
    # Convert matrix to float to avoid dtype conflicts
    logo_matrix = logo_matrix.astype(float) 
    # Scale the matrix by the changes
    for i in range(len(seq)):
        base = seq[i]
        logo_matrix.loc[i, base] *= changes[i] # scale only the original base

    # Create the sequence logo
    logo = lm.Logo(logo_matrix, 
                   color_scheme='classic',
                   figsize=(10,4))

    # Style the plot
    logo.style_spines(visible=False)
    logo.ax.set_xticks(range(len(seq)))
    logo.ax.set_xticklabels(list(seq))
    logo.ax.set_ylabel("Change in Accuracy")
    logo.ax.set_title("Impact of Single-Base Perturbations")
    
    if file_name:
        plt.savefig(file_name, dpi=500, bbox_inches='tight')  # Save as PNG
    plt.close()

def save_checkpoint(self,
                    optimizer, 
                    epoch, 
                    average_loss,  
                    path,):
    """
    Helper function for saving model/optimizer state dicts.
    Only rank 0 should save to avoid file corruption.
    """
    model_state_dict = self.module.state_dict() \
                        if isinstance(self, nn.parallel.DistributedDataParallel) \
                        else self.state_dict()

    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": average_loss,
    }, 
    path,
    )

def main():
    parser = get_argument_parser()
    parser.add_argument(
        "--save_plot_dir",
        type=str,
        default="sequence_visualization",
        required=False
    )
    args = parser.parse_args()
    args_dict = vars(args)
    model = load_model(**args_dict)
    
    model.to(model.device)
    
    train_data_path = args.train_dataset_path
    train_data = pd.read_csv(train_data_path, sep='\t')
    
    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],  # add RNA characters, N is uncertain
        model_max_length=model.max_length,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )
    train_dataset = perturbDataset(
        data=train_data,
        mRNA_max_length=model.mRNA_max_len,
        miRNA_max_length=model.miRNA_max_len, 
        tokenizer=tokenizer,
        use_padding=model.use_padding,
        rc_aug=None,
        add_eos=model.add_eos,
        concat=True,
        add_linker=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=False)

    optimizer = optim.AdamW(
        model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay
    )
    ckpt_path = os.path.join(PROJ_HOME, 
                        "checkpoints", 
                        "TargetScan", 
                        model.model_name, 
                        str(model.mRNA_max_len), 
                        "checkpoint_epoch_final.pth")
    loaded_data = torch.load(ckpt_path, map_location=model.device)
    optimizer.load_state_dict(loaded_data["optimizer_state_dict"])
    
    avg_loss_l = []
    model_checkpoints_dir = os.path.join(
            PROJ_HOME, 
            "checkpoints", 
            model.dataset_name, 
            model.model_name, 
            str(model.mRNA_max_len),
        )
    os.makedirs(model_checkpoints_dir, exist_ok=True)

    start = time.time()
    for epoch in range(model.epochs):
        avg_loss = train(
            model=model,
            train_loader=train_loader,
            epoch=epoch,
            optimizer=optimizer,
            accumulation_step=model.accumulation_step,
            tokenizer=tokenizer,
        )
        avg_loss_l.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            # save model chekcpoints
            checkpoint_path = os.path.join(model_checkpoints_dir,
                                           f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(optimizer,
                            epoch,
                            avg_loss,
                            path=checkpoint_path,
                            )
        cost = time.time() - start
        remain = cost/(epoch + 1) * (model.epochs - epoch - 1) /3600
        print(f'still remain: {remain} hrs.')
    
    # # perturb miRNA sequence
    # deltas = []
    # for pos in range(len(miRNA_seq)):
    #     single_delta = []
    #     # perturb the base 3 times and average the delta
    #     for _ in range(3):
    #         perturbed_miRNA = single_base_perturbation(seq=miRNA_seq, pos=pos)
    #         encoded = encode_seq(
    #             model=model,
    #             tokenizer=tokenizer,
    #             mRNA_seq=mRNA_seq,
    #             miRNA_seq=perturbed_miRNA,
    #         )
    #         pred = predict(model, **encoded)
    #         delta = abs(wt_prob - pred)
    #         single_delta.append(delta)
    #     deltas.append(sum(single_delta)/len(single_delta))
    
    # os.makedirs(args.save_plot_dir, exist_ok=True)
    # file_path = os.path.join(args.save_plot_dir, f"{miRNA_id}.png")
    # viz_sequence(seq=miRNA_seq, # visualize change on the original miRNA seq
    #              changes=deltas,
    #              file_name=file_path)
    
    # # perturb mRNA sequence
    # deltas = []
    # for pos in range(len(mRNA_seq)):
    #     single_delta = []
    #     # perturb the base 3 times and average the delta
    #     for _ in range(3):
    #         perturbed_mRNA = single_base_perturbation(seq=mRNA_seq, pos=pos)
    #         encoded = encode_seq(
    #             model=model,
    #             tokenizer=tokenizer,
    #             mRNA_seq=perturbed_mRNA,
    #             miRNA_seq=miRNA_seq,
    #         )
    #         pred = predict(model, **encoded)
    #         delta = abs(wt_prob - pred)
    #         single_delta.append(delta)
    #     deltas.append(sum(single_delta)/len(single_delta))
    
    # file_path = os.path.join(args.save_plot_dir, f"{mRNA_id}.png")
    # viz_sequence(seq=mRNA_seq, # visualize change on the original mRNA seq
    #              changes=deltas,
    #              file_name=file_path)

if __name__ == '__main__':
    main()