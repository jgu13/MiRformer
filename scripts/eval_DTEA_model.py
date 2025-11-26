import argparse
import os
import torch
from torch.utils.data import DataLoader
# local import
from DTEA_model import DTEA
from Data_pipeline import SpanDataset, CharacterTokenizer
from utils import load_dataset
from Global_parameters import PROJ_HOME

def eval_DTEA(args):
    mirna_max_len = args.mirna_max_len
    mrna_max_len = args.mrna_max_len
    device = args.device
    embed_dim = args.embed_dim
    ff_dim = args.ff_dim
    num_heads = args.num_heads
    num_layers = args.num_layers
    predict_span = args.predict_span
    predict_binding = args.predict_binding
    predict_cleavage = args.predict_cleavage
    use_longformer = args.use_longformer
    ckpt_path = args.ckpt_path
    test_path = args.test_path
    evaluation = True
    batch_size = 32
    
    tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                model_max_length=mrna_max_len,
                                padding_side="right")
    D_test = load_dataset(test_path, sep=',')
    ds_test = SpanDataset(data=D_test,
                        mrna_max_len=mrna_max_len,
                        mirna_max_len=mirna_max_len,
                        tokenizer=tokenizer,
                        seed_start_col="seed start" if "seed start" in D_test.columns else None,
                        seed_end_col="seed end" if "seed end" in D_test.columns else None,
                        cleavage_site_col="cleave_site" if "cleave_site" in D_test.columns else None)
    test_loader = DataLoader(ds_test,
                            batch_size=batch_size, 
                            shuffle=False)

    model = DTEA(
        mirna_max_len=mirna_max_len, 
        mrna_max_len=mrna_max_len, 
        device=device, 
        embed_dim=embed_dim, 
        ff_dim=ff_dim, 
        num_heads=num_heads, 
        num_layers=num_layers, 
        predict_span=predict_span,
        predict_binding=predict_binding,
        predict_cleavage=predict_cleavage,
        use_longformer=use_longformer
    )
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.to(device)
    print(f"Loaded checkpoint from {ckpt_path}")
    print("Evaluating model...")
    model.eval_loop(model=model, dataloader=test_loader, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirna_max_len", type=int, default=24)
    parser.add_argument("--mrna_max_len", type=int, default=520)
    parser.add_argument("--ckpt_path", type=str, 
            default=os.path.join(PROJ_HOME, 
            "checkpoints/TargetScan/TwoTowerTransformer/Longformer/520/embed=1024d/norm_by_key/LSE/best_composite_0.9112_0.9922_epoch28.pth"),
            help="Path to the checkpoint file.")
    parser.add_argument("--test_path", type=str, help="Path to the test data file.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--ff_dim", type=int, default=4096)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--predict_span", action='store_true') # store true when the flag is set
    parser.add_argument("--predict_binding", action='store_true') # store true when the flag is set
    parser.add_argument("--predict_cleavage", action='store_true') # store true when the flag is set
    parser.add_argument("--use_longformer", action='store_true')

    args = parser.parse_args()
    eval_DTEA(args)