import os
import torch
import random
import time
import numpy as np
import pandas as pd
import logomaker as lm
import seaborn as sns
import matplotlib.pyplot as plt
# local import
from mirLM import mirLM
from argument_parser import get_argument_parser
from Data_pipeline import CharacterTokenizer

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

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
        elif model.base_model_name == 'TwoTowerMLP':
            mRNA_seq = kwargs["mRNA_seq"]
            miRNA_seq = kwargs["miRNA_seq"]
            mRNA_seq_mask = kwargs["mRNA_seq_mask"]
            miRNA_seq_mask = kwargs["miRNA_seq_mask"]
            return_attn = kwargs["return_attn"]
            output, attn_weights = model(
                mRNA_seq,
                miRNA_seq,
                mRNA_seq_mask=mRNA_seq_mask,
                miRNA_seq_mask=miRNA_seq_mask,
                return_attn=return_attn
            )
        prob = torch.sigmoid(output.squeeze()).item()
    if kwargs.get("return_attn", ""):
        return prob, attn_weights
    else:
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
    elif model.base_model_name == "TwoTowerMLP":
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
                            "checkpoint_epoch_69.pth")
    loaded_data = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(loaded_data["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")
    return model


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
    test_data = pd.read_csv(test_data_path, sep=',')
    mRNA_seqs = test_data[["mRNA sequence"]].values
    miRNA_seqs = test_data[["miRNA sequence"]].values
    labels = test_data[["label"]].values
    
    # Testing the first sequence
    i=10
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
    
    output, attn_weights = predict(
        model=model,
        mRNA_seq=encoded["mRNA_seq"],
        miRNA_seq=encoded["miRNA_seq"],
        mRNA_seq_mask=encoded["mRNA_seq_mask"],
        miRNA_seq_mask=encoded["miRNA_seq_mask"],
        return_attn=True,
    )
    print("Attn_Weights shape = ", attn_weights.shape)
    # Select the first sample (or any sample index you prefer)
    sample_attn = attn_weights[0].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(sample_attn, 
                cmap="Blues", 
                xticklabels=list(mRNA_seq), 
                yticklabels=list(miRNA_seq))
    plt.xlabel("mRNA bases")
    plt.ylabel("miRNA bases")
    plt.title("Cross-Attention Heatmap with Base Labels")
    file_name = os.path.join(args_dict["save_plot_dir"], f"{mRNA_id}_{miRNA_id}_heatmap.png")
    print(f"Saved plot to {file_name}")
    plt.savefig(file_name, dpi=800, bbox_inches='tight')
    
if __name__=="__main__":
    main()