import argparse
import os
from typing import Dict, Iterable, Tuple

import torch
import pandas as pd

import transformer_model as tm
from Data_pipeline import CharacterTokenizer


PROJ_HOME = os.path.expanduser("~/projects/mirLM")


def load_model(ckpt_path: str, model_args: Dict) -> tm.QuestionAnsweringModel:
    model = tm.QuestionAnsweringModel(**model_args)
    state_dict = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(state_dict, strict=False)
    model.to(model_args["device"])
    model.eval()
    return model


def build_tokenizer(model: tm.QuestionAnsweringModel) -> CharacterTokenizer:
    return CharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],
        model_max_length=model.mrna_max_len,
        add_special_tokens=False,
        padding_side="right",
    )


def encode_sequences(
    tokenizer: CharacterTokenizer,
    model: tm.QuestionAnsweringModel,
    mrna_seq: str,
    mirna_seq: str,
) -> Dict[str, torch.Tensor]:
    mrna_encoding = tokenizer(
        mrna_seq,
        add_special_tokens=False,
        padding="max_length",
        max_length=model.mrna_max_len,
        truncation=True,
        return_attention_mask=True,
    )
    mirna_encoding = tokenizer(
        mirna_seq,
        add_special_tokens=False,
        padding="max_length",
        max_length=model.mirna_max_len,
        truncation=True,
        return_attention_mask=True,
    )
    encoded = {
        "mRNA_seq": torch.tensor(
            mrna_encoding["input_ids"], dtype=torch.long, device=model.device
        ).unsqueeze(0),
        "miRNA_seq": torch.tensor(
            mirna_encoding["input_ids"], dtype=torch.long, device=model.device
        ).unsqueeze(0),
        "mRNA_seq_mask": torch.tensor(
            mrna_encoding["attention_mask"], dtype=torch.long, device=model.device
        ).unsqueeze(0),
        "miRNA_seq_mask": torch.tensor(
            mirna_encoding["attention_mask"], dtype=torch.long, device=model.device
        ).unsqueeze(0),
    }
    return encoded


def compute_attention_snapshot(
    model: tm.QuestionAnsweringModel, batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    with torch.no_grad():
        _ = model(
            mirna=batch["miRNA_seq"],
            mrna=batch["mRNA_seq"],
            mirna_mask=batch["miRNA_seq_mask"],
            mrna_mask=batch["mRNA_seq_mask"],
        )
    attn = model.predictor.cross_attn_layer.last_attention
    return torch.amax(attn[0], dim=0).detach().cpu()  # (mrna, mirna)


def split_seed_regions(
    attn_matrix: torch.Tensor, seed_start: int, seed_end: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    seed_weights = attn_matrix[seed_start:seed_end, :]
    non_seed_blocks = [attn_matrix[0:seed_start, :], attn_matrix[seed_end + 1 :, :]]
    non_seed_weights = torch.cat(non_seed_blocks, dim=0)
    return seed_weights, non_seed_weights


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)


def save_snapshot(
    seed_weights: torch.Tensor,
    non_seed_weights: torch.Tensor,
    meta: Dict,
    out_dir: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{safe_name(meta['miRNA_id'])}__{safe_name(meta['mRNA_id'])}__row{meta['row_idx']}.pt"
    path = os.path.join(out_dir, fname)
    payload = {
        "seed_weights": seed_weights.cpu(),
        "non_seed_weights": non_seed_weights.cpu(),
        "meta": meta,
    }
    torch.save(payload, path)
    return path


def get_default_indices(arg: str) -> Iterable[int]:
    return [int(x) for x in arg.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache seed and non-seed attention weights for later plotting."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(
            PROJ_HOME,
            "TargetScan_dataset",
            "TargetScan_train_500_randomized_start.csv",
        ),
        help="CSV containing sequences.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(
            PROJ_HOME,
            "checkpoints",
            "TargetScan",
            "TwoTowerTransformer",
            "Longformer",
            "520",
            "mean_unchunk_best_composite_0.7690_0.9038_epoch15.pth",
        ),
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--indices",
        type=get_default_indices,
        default=get_default_indices("8,12,24,48"),
        help="Comma-separated row indices to process.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            PROJ_HOME,
            "Performance",
            "TargetScan_test",
            "TwoTowerTransformer",
            "attention_snapshots",
        ),
        help="Directory to write tensor snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pd.read_csv(args.dataset)
    model_args = {
        "mirna_max_len": 24,
        "mrna_max_len": 520,
        "device": args.device,
        "embed_dim": 1024,
        "num_heads": 8,
        "num_layers": 4,
        "ff_dim": 2048,
        "predict_span": True,
        "predict_binding": True,
        "use_longformer": False,
    }
    model = load_model(args.checkpoint, model_args)
    tokenizer = build_tokenizer(model)

    saved_paths = []
    for idx in args.indices:
        row = dataset.iloc[idx]
        mrna_seq = row["mRNA sequence"]
        mirna_seq = row["miRNA sequence"].replace("U", "T")[::-1]
        meta = {
            "row_idx": int(idx),
            "mRNA_id": row["Transcript ID"],
            "miRNA_id": row["miRNA ID"],
            "seed_start": int(row["seed start"]),
            "seed_end": int(row["seed end"]),
        }
        batch = encode_sequences(tokenizer, model, mrna_seq, mirna_seq)
        attn = compute_attention_snapshot(model, batch)
        seed_weights, non_seed_weights = split_seed_regions(
            attn, meta["seed_start"], meta["seed_end"]
        )
        path = save_snapshot(seed_weights, non_seed_weights, meta, args.output_dir)
        saved_paths.append(path)
        print(f"Saved attention snapshot for row {idx} -> {path}")

    print(f"Done. Wrote {len(saved_paths)} attention subsets to {args.output_dir}")


if __name__ == "__main__":
    main()
