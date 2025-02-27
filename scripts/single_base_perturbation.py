import os
import torch
import random
import time
import numpy as np
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt
# local import
from mirLM import mirLM
from argument_parser import get_argument_parser
from Data_pipeline import CharacterTokenizer

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")

def predict(model, 
            **kwargs
            ):
    model.to(model.device)
    model.eval()
    with torch.no_grad():
        if model.base_model_name == 'HyenaDNA':
            seq = kwargs["seq"]
            seq_mask = kwargs["seq_mask"]
            output = model(seq, seq_mask, perturb=False)
        elif model.base_model_name == 'TwoTower':
            mRNA_seq = kwargs["mRNA_seq"]
            miRNA_seq = kwargs["miRNA_seq"]
            mRNA_seq_mask = kwargs["mRNA_seq_mask"]
            miRNA_seq_mask = kwargs["miRNA_seq_mask"]
            output = model(
                mRNA_seq,
                miRNA_seq,
                mRNA_seq_mask=mRNA_seq_mask,
                miRNA_seq_mask=miRNA_seq_mask,
            )
        prob = torch.sigmoid(output.squeeze()).item()
    return prob

def encode_seq(model, 
               tokenizer, 
               mRNA_seq, 
               miRNA_seq,
               ):
    mRNA_seq_encoding = tokenizer(
        mRNA_seq,
        add_special_tokens=False,
        padding="max_length" if model.use_padding else None,
        max_length=model.mRNA_max_len,
        truncation=True,
        return_attention_mask=True,
    )
    mRNA_seq_tokens = mRNA_seq_encoding["input_ids"]  # get input_ids
    mRNA_seq_mask = mRNA_seq_encoding["attention_mask"]  # get attention mask
    # print("Tokenized sequence length = ", len(mRNA_seq_tokens))

    miRNA_seq_encoding = tokenizer(
        miRNA_seq,
        add_special_tokens=False,
        padding="max_length" if model.use_padding else None,
        max_length=model.miRNA_max_len,
        truncation=True,
        return_attention_mask=True,
    )
    miRNA_seq_tokens = miRNA_seq_encoding["input_ids"]  # get input_ids
    miRNA_seq_mask = miRNA_seq_encoding["attention_mask"]  # get attention mask
    
    if model.base_model_name == "HyenaDNA":
        linker = [tokenizer._convert_token_to_id("N")] * 6
        concat_seq_tokens = mRNA_seq_tokens + linker + miRNA_seq_tokens
        linker_mask = [1] * 6
        concat_seq_mask = mRNA_seq_mask + linker_mask + miRNA_seq_mask
        # convert to tensor
        concat_seq_tokens = torch.tensor(concat_seq_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
        concat_seq_mask = torch.tensor(concat_seq_mask, dtype=torch.long, device=model.device).unsqueeze(0)
        return {"seq": concat_seq_tokens, 
                "seq_mask": concat_seq_mask}
    elif model.base_model_name == "TwoTower":
        mRNA_seq_tokens = torch.tensor(mRNA_seq_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
        miRNA_seq_tokens = torch.tensor(miRNA_seq_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
        mRNA_seq_mask = torch.tensor(mRNA_seq_mask, dtype=torch.long, device=model.device).unsqueeze(0)
        miRNA_seq_mask = torch.tensor(miRNA_seq_mask, dtype=torch.long, device=model.device).unsqueeze(0)
        return {"mRNA_seq": mRNA_seq_tokens, 
                "miRNA_seq": miRNA_seq_tokens, 
                "mRNA_seq_mask": mRNA_seq_mask, 
                "miRNA_seq_mask": miRNA_seq_mask}
    
def load_model(**args_dict):
    # load model checkpoint
    model = mirLM.create_model(**args_dict)
    ckpt_path = os.path.join(PROJ_HOME, 
                            "checkpoints", 
                            model.dataset_name, 
                            model.model_name, 
                            str(model.mRNA_max_len), 
                            "checkpoint_epoch_19.pth")
    loaded_data = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(loaded_data["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")
    return model

def single_base_perturbation(seq, pos):
    bases = ['A','T','C','G']
    seq = list(seq)
    orig = seq[pos]
    seq[pos] = random.choice([base for base in bases if base != orig])
    return ''.join(seq)

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
                   figsize=(10,2))

    # Style the plot
    logo.style_spines(visible=False)
    logo.ax.set_xticks(range(len(seq)))
    logo.ax.set_xticklabels(list(seq))
    # logo.ax.set_yscale('log')  # Log-transformed y-axis
    # logo.ax.set_ylim(1e-6, 1e-5)  # Example: 1e-5 to 0.1
    logo.ax.set_ylabel("Change in Accuracy")
    logo.ax.set_title("Impact of Single-Base Perturbations")
    
    if file_name:
        plt.savefig(file_name, dpi=500, bbox_inches='tight')  # Save as PNG
    plt.close()

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
    
    test_data_path = args.test_dataset_path
    test_data = pd.read_csv(test_data_path)
    mRNA_seqs = test_data[["mRNA sequence"]].values
    miRNA_seqs = test_data[["miRNA sequence"]].values
    labels = test_data[["label"]].values
    
    # Testing the first sequence
    i=1899
    mRNA_seq = mRNA_seqs[i][0]
    miRNA_seq = miRNA_seqs[i][0]
    miRNA_id = test_data[["miRNA ID"]].iloc[i,0]
    mRNA_id = test_data[["Gene Symbol"]].iloc[i,0]
    print("miRNA id = ",miRNA_id)
    print("mRNA id = ", mRNA_id)
    # replace U with T and reverse miRNA to 3' to 5' 
    miRNA_seq = miRNA_seq.replace("U", "T")[::-1]
    
    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],  # add RNA characters, N is uncertain
        model_max_length=model.max_length,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )
    encoded = encode_seq(
        model=model, 
        tokenizer=tokenizer,
        mRNA_seq=mRNA_seq,
        miRNA_seq=miRNA_seq,
    )
    wt_prob = predict(model, **encoded)
    print("Wild-type prediction = ", wt_prob)
    
    # perturb miRNA sequence
    deltas = []
    for pos in range(len(miRNA_seq)):
        single_delta = []
        # perturb the base 3 times and average the delta
        for _ in range(3):
            perturbed_miRNA = single_base_perturbation(seq=miRNA_seq, pos=pos)
            encoded = encode_seq(
                model=model,
                tokenizer=tokenizer,
                mRNA_seq=mRNA_seq,
                miRNA_seq=perturbed_miRNA,
            )
            pred = predict(model, **encoded)
            delta = abs(wt_prob - pred)
            single_delta.append(delta)
        deltas.append(sum(single_delta)/len(single_delta))
    
    os.makedirs(args.save_plot_dir, exist_ok=True)
    file_path = os.path.join(args.save_plot_dir, f"{miRNA_id}.png")
    viz_sequence(seq=miRNA_seq, # visualize change on the original miRNA seq
                 changes=deltas,
                 file_name=file_path)
    
    # perturb mRNA sequence
    deltas = []
    for pos in range(len(mRNA_seq)):
        single_delta = []
        # perturb the base 3 times and average the delta
        for _ in range(3):
            perturbed_mRNA = single_base_perturbation(seq=mRNA_seq, pos=pos)
            encoded = encode_seq(
                model=model,
                tokenizer=tokenizer,
                mRNA_seq=perturbed_mRNA,
                miRNA_seq=miRNA_seq,
            )
            pred = predict(model, **encoded)
            delta = abs(wt_prob - pred)
            single_delta.append(delta)
        deltas.append(sum(single_delta)/len(single_delta))
    
    file_path = os.path.join(args.save_plot_dir, f"{mRNA_id}.png")
    viz_sequence(seq=mRNA_seq, # visualize change on the original mRNA seq
                 changes=deltas,
                 file_name=file_path)

if __name__ == '__main__':
    main()