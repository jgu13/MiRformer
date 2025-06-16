import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import transformer_model as tm
from Data_pipeline import CharacterTokenizer

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

def load_model(
        ckpt_name,
        **args_dict):
    # load model checkpoint
    model = tm.QuestionAnsweringModel(**args_dict)
    ckpt_path = os.path.join(PROJ_HOME, 
                            "checkpoints", 
                            "TargetScan/TwoTowerTransformer",
                            str(model.mrna_max_len), 
                            ckpt_name)
    loaded_data = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(loaded_data)
    print(f"Loaded checkpoint from {ckpt_path}")
    return model

def encode_seq(model, 
               tokenizer, 
               mRNA_seq, 
               miRNA_seq,
               ):
    mRNA_seq_encoding = tokenizer(
        mRNA_seq,
        add_special_tokens=False,
        padding="max_length",
        max_length=model.mrna_max_len,
        truncation=True,
        return_attention_mask=True,
    )
    mRNA_seq_tokens = mRNA_seq_encoding["input_ids"]  # get input_ids
    mRNA_seq_mask = mRNA_seq_encoding["attention_mask"]  # get attention mask

    miRNA_seq_encoding = tokenizer(
        miRNA_seq,
        add_special_tokens=False,
        padding="max_length",
        max_length=model.mirna_max_len,
        truncation=True,
        return_attention_mask=True,
    )
    miRNA_seq_tokens = miRNA_seq_encoding["input_ids"]  # get input_ids
    miRNA_seq_mask = miRNA_seq_encoding["attention_mask"]  # get attention mask

    mRNA_seq_tokens = torch.tensor(mRNA_seq_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
    miRNA_seq_tokens = torch.tensor(miRNA_seq_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
    mRNA_seq_mask = torch.tensor(mRNA_seq_mask, dtype=torch.long, device=model.device).unsqueeze(0)
    miRNA_seq_mask = torch.tensor(miRNA_seq_mask, dtype=torch.long, device=model.device).unsqueeze(0)
    return {"mRNA_seq": mRNA_seq_tokens, 
            "miRNA_seq": miRNA_seq_tokens, 
            "mRNA_seq_mask": mRNA_seq_mask, 
            "miRNA_seq_mask": miRNA_seq_mask}

def predict(model,
            mRNA_seq,
            miRNA_seq,
            mRNA_seq_mask,
            miRNA_seq_mask,):
    model.eval()
    with torch.no_grad():
        print("mRNA_seq shape: ", mRNA_seq.shape)
        print("miRNA_seq shape: ", miRNA_seq.shape)
        _ = model(
            mirna = mRNA_seq,
            mrna = miRNA_seq,
            mirna_mask = miRNA_seq_mask,
            mrna_mask = mRNA_seq_mask
        )

def plot_heatmap(model,
                 mRNA_ids,
                 miRNA_ids,
                 save_plot_dir):
    attn_weights = model.predictor.cross_attn_layer.last_attention
    attn_weights = attn_weights[0]

    # plot attn weights for each head
    for h in range(attn_weights.size[0]):
        head_weights = attn_weights[h]
        sns.heatmap(head_weights, 
                cmap="Blues", 
                xticklabels=mRNA_ids, 
                yticklabels=miRNA_ids)
        plt.xlabel("mRNA bases")
        plt.ylabel("miRNA bases")
        plt.title("Cross-Attention Heatmap with Base Labels")
        file_name = os.path.join(save_plot_dir, "heatmap.png")
        print(f"Saved plot to {file_name}")
        plt.savefig(file_name, dpi=800, bbox_inches='tight')
        print(f"Heatmap is saved to {file_name}")

def main():
    mirna_max_len = 24
    mrna_max_len  = 30
    device        = "cuda:1"
    predict_span  = False
    predict_binding = True 
    args_dict = {"mirna_max_len": mirna_max_len,
                 "mrna_max_len": mrna_max_len,
                 "device": device,
                 "predict_span": predict_span,
                 "predict_binding": predict_binding,}
    data_dir = os.path.join(PROJ_HOME, 'TargetScan_dataset')
    test_datapath = os.path.join(PROJ_HOME, data_dir, "TargetScan_test_30_randomized_start.csv")
    test_data = pd.read_csv(test_datapath, sep=',')
    mRNA_seqs = test_data[["mRNA sequence"]].values
    miRNA_seqs = test_data[["miRNA sequence"]].values
    
    model = load_model(ckpt_name="best_binding_acc_0.9959_epoch77.pth",
                       **args_dict)

    # Testing the first sequence
    i=0
    mRNA_seq = mRNA_seqs[i][0]
    miRNA_seq = miRNA_seqs[i][0]
    # replace U with T and reverse miRNA to 3' to 5' 
    miRNA_seq = miRNA_seq.replace("U", "T")[::-1]
    
    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],  # add RNA characters, N is uncertain
        model_max_length=model.mrna_max_len,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="right",  # since HyenaDNA is causal, we pad on the left
    )
    encoded = encode_seq(
        model=model, 
        tokenizer=tokenizer,
        mRNA_seq=mRNA_seq,
        miRNA_seq=miRNA_seq,
    )
    # convert mRNA seq tokens to mRNA ids
    mRNA_tokens = encoded["mRNA_seq"].squeeze(0)
    mRNA_ids = [tokenizer._convert_id_to_token(d.item()) for d in mRNA_tokens]
    # convert miRNA seq tokens to mRNA ids
    miRNA_tokens = encoded["miRNA_seq"].squeeze(0)
    miRNA_ids = [tokenizer._convert_id_to_token(d.item()) for d in miRNA_tokens]

    model.to(args_dict["device"])
    predict(model=model,
            mRNA_seq=mRNA_tokens,
            miRNA_seq=miRNA_tokens,
            mRNA_seq_mask=encoded["mRNA_seq_mask"],
            miRNA_seq_mask=encoded["miRNA_seq_mask"],
    )

    save_plot_dir = os.path.join(PROJ_HOME, "Performance/TargetScan_test", "TwoTowerTransformer")
    os.makedirs(save_plot_dir, exist_ok=True)
    plot_heatmap(model,
                 miRNA_ids=miRNA_ids,
                 mRNA_ids=mRNA_ids,
                 save_plot_dir=save_plot_dir)
    
if __name__ == '__main__':
    main()