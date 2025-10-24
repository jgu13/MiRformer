import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import pandas as pd
import numpy as np
from utils import load_dataset
from Data_pipeline import QuestionAnswerDataset
from Data_pipeline import CharacterTokenizer
from transformer_model import QuestionAnsweringModel

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")

from torch.nn.utils.rnn import pad_sequence

def make_qa_collate(tokenizer):
    # pick a pad id. Since you added "N" to the alphabet, use it.
    # This is robust even for custom tokenizers:
    pad_id = tokenizer._convert_token_to_id("[PAD]")

    def _round_up(x, m):
        return ((x + m - 1) // m) * m

    def _pad_to_multiple(batch_padded_2d, multiple, pad_value):
        # batch_padded_2d: [B, Lmax] after pad_sequence
        B, L = batch_padded_2d.shape
        L_rounded = _round_up(L, multiple)
        if L_rounded == L:
            return batch_padded_2d
        # right-pad along seq dim
        return F.pad(batch_padded_2d, (0, L_rounded - L), value=pad_value)

    def collate_fn(batch):
        # sequences
        mrna_list  = [b["mrna_input_ids"]  for b in batch]
        mirna_list = [b["mirna_input_ids"] for b in batch]

        # pad to batch max length
        mrna_padded  = pad_sequence(mrna_list,  batch_first=True, padding_value=pad_id)
        mirna_padded = pad_sequence(mirna_list, batch_first=True, padding_value=pad_id)

        # round mrna lengths up to multiple=40 (independently for mRNA)
        mrna_padded  = _pad_to_multiple(mrna_padded,  multiple=40, pad_value=pad_id)

        # attention masks: 1 for real tokens, 0 for pad
        mrna_mask  = (mrna_padded  != pad_id).long()
        mirna_mask = (mirna_padded != pad_id).long()

        # scalar labels
        start_pos = torch.stack([torch.as_tensor(b["start_positions"], dtype=torch.long) for b in batch])
        end_pos   = torch.stack([torch.as_tensor(b["end_positions"],   dtype=torch.long) for b in batch])
        targets   = torch.stack([torch.as_tensor(b["target"],          dtype=torch.long) for b in batch]).view(-1)
        cleaves   = torch.stack([torch.as_tensor(b["cleavage_sites"],  dtype=torch.long) for b in batch])

        return {
            "mrna_input_ids": mrna_padded,
            "mrna_attention_mask": mrna_mask,
            "mirna_input_ids": mirna_padded,
            "mirna_attention_mask": mirna_mask,
            "start_positions": start_pos,
            "end_positions": end_pos,
            "target": targets,            # shape [B]
            "cleavage_sites": cleaves
        }
    return collate_fn


def prepare_dataset(model, data_path):
    D = load_dataset(data_path, sep='\t' if data_path.endswith('.tsv') else ',')
    model.D = D
    tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                    model_max_length=model.mrna_max_len,
                                    padding_side="right")
    ds = QuestionAnswerDataset(data=D,
                                tokenizer=tokenizer,
                                seed_start_col=None,
                                seed_end_col=None, # do not load seed start and end positions
                                cleavage_site_col="cleave_site")
    ds_loader   = DataLoader(ds, 
                        batch_size=model.batch_size,
                        shuffle=False,
                        collate_fn=make_qa_collate(tokenizer))  # <-- important)
    return ds_loader

def load_model(model,ckpt_path):
    #skip loading RotaryEmbedding weights
    loaded_state_dict = torch.load(ckpt_path, map_location=model.device)
    filtered_state_dict = {}
    for key, value in loaded_state_dict.items():
        if "rotary" in key:
            print(f"Skipping {key} from {ckpt_path}", flush=True)
            continue
        filtered_state_dict[key] = value

    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Loaded {len(filtered_state_dict)} weights from {ckpt_path}", flush=True)

def predict_loop(model, dataloader, device, W_list=[3,5]):
        model.eval()
        all_start_preds, all_end_preds        = [], []
        all_cleavage_preds, all_cleavage_labels = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                for k in batch:
                    batch[k] = batch[k].to(device)
                mrna_mask  = batch["mrna_attention_mask"].to(device)
                mirna_mask = batch["mirna_attention_mask"].to(device)
                mirna_ids = batch["mirna_input_ids"].to(device)
                mrna_ids = batch["mrna_input_ids"].to(device)
                outputs    = model(
                    mirna=mirna_ids,
                    mrna=mrna_ids,
                    mrna_mask=mrna_mask,
                    mirna_mask=mirna_mask,
                )

                binding_logit, binding_weights, start_logits, end_logits, cleavage_logits = outputs

                # Cleavage site prediction loss and metrics
                cleavage_loss = torch.tensor(0.0, device=device)
                if model.predict_cleavage: 
                    cleavage_targets = batch["cleavage_sites"]  # (batchsize,)
                    # Cleavage predictions
                    cleavage_preds = torch.argmax(cleavage_logits, dim=-1)
                    all_cleavage_preds.extend(cleavage_preds.cpu())
                    all_cleavage_labels.extend(cleavage_targets.cpu())

                # span loss and predictions
                if model.predict_span and start_logits is not None and end_logits is not None:
                    # predictions
                    start_preds = torch.argmax(start_logits, dim=-1) #(batch_size, )
                    end_preds   = torch.argmax(end_logits, dim=-1) #(batch_size, )
                    all_start_preds.extend(start_preds.cpu())
                    all_end_preds.extend(end_preds.cpu())
                
                # save per batch predictions
                if model.predict_span:
                    model.all_start_preds = np.array(all_start_preds)
                    model.all_end_preds = np.array(all_end_preds)
                if len(all_cleavage_preds) > 0:
                    model.all_cleavage_preds = np.array(all_cleavage_preds)
                # save as pandas dataframe where columns are "pred start", "pred end", "pred cleavage"    
                df = pd.DataFrame({
                    "pred start": model.all_start_preds,
                    "pred end": model.all_end_preds,
                    "pred cleavage": model.all_cleavage_preds
                })
                df.to_csv(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data", f"predicted_seeds_and_cleave_sites_predictions_batch_{batch_idx}.csv"), index=False)

        # if there are positive examples
        if len(all_start_preds) > 0:
            all_start_preds  = torch.stack(all_start_preds).detach().cpu().long()
            all_end_preds    = torch.stack(all_end_preds).detach().cpu().long()

        # Cleavage site accuracy
        if len(all_cleavage_preds) > 0:
            all_cleavage_preds  = torch.tensor(all_cleavage_preds, dtype=torch.long)
            all_cleavage_labels = torch.tensor(all_cleavage_labels, dtype=torch.long)
            acc_cleavage        = (all_cleavage_preds == all_cleavage_labels).float().mean().item()
            if W_list is not None:
                hit_at_w_list = {}
                for w in W_list:
                    hit_at_w = ((all_cleavage_preds - all_cleavage_labels).abs() <= w).float().mean().item()
                    hit_at_w_list[f"Hit at {w}"] = hit_at_w
            else:
                hit_at_w_list   = None
        else:
            acc_cleavage        = 0.0
            hit_at_w_list       = None
        
        # save per batch prediction
        if model.predict_span:
            model.all_start_preds = all_start_preds.numpy()
            model.all_end_preds = all_end_preds.numpy()
        if len(all_cleavage_preds) > 0:
            model.all_cleavage_preds = all_cleavage_preds.numpy()
        # save as pandas dataframe where columns are "pred start", "pred end", "pred cleavage"    
        df = pd.DataFrame({
            "pred start": model.all_start_preds,
            "pred end": model.all_end_preds,
            "pred cleavage": model.all_cleavage_preds
        })
        df.to_csv(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data", "predicted_seeds_and_cleave_sites_predictions.csv"), index=False)
        
        print(f"Cleavage Acc: {acc_cleavage*100}")
        if hit_at_w_list is not None:
            for w, hit_at_w in hit_at_w_list.items():
                print(f"{w}: {hit_at_w*100}%")

        return acc_cleavage, hit_at_w_list 

def predict_seeds_and_cleave_sites(model, dataloader, save_path=None):
    acc_cleavage, hit_at_w_list = predict_loop(model=model,
                                                dataloader=dataloader,
                                                device=model.device,
                                                W_list=[3,5])
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "predicted_seeds_and_cleave_sites_metrics.txt"), "w") as f:
            f.write(f"Cleavage accuracy: {acc_cleavage}\n")
            if hit_at_w_list is not None:
                for w, hit_at_w in hit_at_w_list.items():
                    f.write(f"{w}: {hit_at_w}\n")
        print(f"Metrics saved to {save_path}", flush=True)

def save_results(model, prediction_path=None):
    D_test_positive = model.D.copy()
    if model.predict_span:
        D_test_positive["pred start"] = model.all_start_preds
        D_test_positive["pred end"]   = model.all_end_preds
    if model.predict_cleavage:
        D_test_positive["pred cleavage"] = model.all_cleavage_preds
    
    pred_df_path = os.path.join(os.path.join(prediction_path, "seed_span_predictions.csv"))
    os.makedirs(prediction_path, exist_ok=True)
    D_test_positive.to_csv(pred_df_path, index=False)
    print(f"Prediction saved to {prediction_path}")


def main():
    torch.cuda.empty_cache() # clear crashed cache
    mrna_max_len = 25000 # max length of UTR sequence
    mirna_max_len = 30
    test_datapath  = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/starbase_degradome_UTR.tsv")
    ckpt_path = os.path.join(PROJ_HOME, "checkpoints/TargetScan/TwoTowerTransformer/Longformer/520/predict_cleavage/continue_training/UTR_windows_500/gaussian_smoothed/continue_training_best_composite_0.9042_0.9871_epoch12_best_acc_and_hit_1.3789_epoch42.pth")
    metrics_path = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data")
    prediction_path = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data")
    model = QuestionAnsweringModel(mrna_max_len=mrna_max_len,
                                   mirna_max_len=mirna_max_len,
                                   device="cuda:1",
                                   epochs=50,
                                   embed_dim=1024,
                                   num_heads=8,
                                   num_layers=4,
                                   ff_dim=4096,
                                   batch_size=2,
                                   lr=3e-5,
                                   seed=10020,
                                   predict_span=True,
                                   predict_binding=False,
                                   predict_cleavage=True,
                                   use_longformer=True)
    data_loader = prepare_dataset(model, test_datapath)
    load_model(model, ckpt_path)
    model.to(model.device)
    predict_seeds_and_cleave_sites(model, data_loader, save_path=metrics_path)
    save_results(model, prediction_path=prediction_path)

if __name__ == "__main__":
    main()