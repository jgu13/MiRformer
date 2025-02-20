import os
import shap
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from deeplift.visualization import viz_sequence
from deeplift.dinuc_shuffle import dinuc_shuffle
# local imports
from baseline_CNN import BaselineCNN
from Data_pipeline import CharacterTokenizer, miRawDataset

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

def load_model(model_name, 
               **kwargs):
    from TwoTowerMLP import TwoTowerMLP
    from HyenaDNAWrapper import HyenaDNAWrapper
    
    if model_name == 'CNN':
        model = BaselineCNN(**kwargs)
    elif model_name == 'TwoTowerMLP':
        model = TwoTowerMLP(**kwargs)
    elif model_name == 'HyenaDNA':
        model = HyenaDNAWrapper(**kwargs)
    elif model_name == 'TwoTowerMLP_Attn':
        model = TwoTowerMLP(**kwargs)
    elif model_name == 'Attn':
        model = HyenaDNAWrapper(**kwargs)
    else:
        raise ValueError(f"Unsupported or missing model_name: {model_name}")
    
    return model

def simple_shuffle(seq_str):
    """
    Randomly shuffles the characters in a string.
    
    Args:
        seq_str (str): A sequence string (e.g. "ACTGACTG").
        
    Returns:
        str: The shuffled string.
    """
    seq_list = list(seq_str)
    np.random.shuffle(seq_list)
    return "".join(seq_list)

def shuffle_several_times(s):
    s = np.squeeze(s)
    return dinuc_shuffle(s, num_shufs=100)

# Prepare the background function using dinuc_shuffle
def background_function(s, tokenizer):
    """
    Generate background sequences by randomly shuffling each input sequence.
    
    Args:
        s: A numpy array of shape (N, seq_len) containing integer-encoded sequences.
        tokenizer: A CharacterTokenizer instance.
        
    Returns:
        numpy.ndarray: An array of background sequences in the same tokenized (integer) format.
    """
    background_list = []
    for seq in s:
        # Convert integer sequence to string
        seq_str = int_seq_to_str(seq, tokenizer)
        # Shuffle the string randomly (does not preserve dinucleotide frequencies)
        shuffled_str = simple_shuffle(seq_str)
        # Tokenize the shuffled sequence back to integer ids.
        # Note: Adjust padding and max_length as needed.
        encoding = tokenizer(
            shuffled_str,
            add_special_tokens=False,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=False
        )
        background_list.append(encoding["input_ids"])
    return np.array(background_list)

def int_seq_to_str(seq, tokenizer):
    """
    Convert an integer-encoded sequence (1D array) to a string using the tokenizer.
    Assumes that tokenizer._convert_id_to_token(x) returns the string for token x.
    """
    # Ensure each token is an integer
    return "".join([tokenizer._convert_id_to_token(int(x)) for x in seq])

def tokens_to_basewise_shap(token_ids, shap_values):
    """
    Project shap_values (shape=(seq_len,)) onto the actual base at each position.
    We'll produce an array of shape (4, seq_len) for A/C/G/T channels, so that
    position i has shap only at the channel of the base that actually appears.
    
    token_ids: 1D array of length seq_len, containing integer-coded tokens:
                7=A, 8=C, 9=T, 10=G, etc.
    shap_values: 1D array of length seq_len with SHAP importance per position.
    returns: (4, seq_len) float array for A, C, G, T channels
    """
    seq_len = token_ids.shape[0]
    projected = np.zeros((4, seq_len), dtype=np.float32)

    for i in range(seq_len):
        tok = token_ids[i]
        sv = shap_values[i]  # shap value at this position
        # map token to channel
        if tok == 7:   # A
            projected[0, i] = sv
        elif tok == 8: # C
            projected[1, i] = sv
        elif tok == 9: # G
            projected[2, i] = sv
        elif tok == 10:  # T
            projected[3, i] = sv
        else:
            # e.g. [PAD], [UNK], or 'N'.
            pass
    
    return projected

def evaluate_and_explain(model, test_loader, device="cuda", n_background=50):
    """
    Evaluate a PyTorch model that takes integer-encoded tokens
    and compute SHAP values at each sequence position.

    Args:
        model: a PyTorch model whose forward() takes shape (batch, seq_len) integers
        test_loader: DataLoader producing (token_ids, label) or (token_ids, mask, label) ...
        device: "cuda" or "cpu"
        n_background: number of random samples from the test set to use as background

    Returns:
        shap_values: A list (if multi-class) or an array (if single output) of SHAP arrays.
                     Typically shape = [n_samples, seq_len].
        predictions: The model's predicted classes (or probabilities).
    """

    model.eval().to(device)

    # 1) Gather all test samples
    if model.model_name == "HyenaDNA":
        with torch.no_grad():
            all_tokens = []
            all_labels = []
            all_masks = []
            for (seq, seq_mask, target) in test_loader:
                all_tokens.append(seq.cpu().numpy())  # shape: (B, seq_len)
                all_masks.append(seq_mask.cpu().numpy())
                all_labels.append(target.cpu().numpy())
            X_test = np.concatenate(all_tokens, axis=0)  # shape: (N, seq_len)
            X_test_mask = np.concatenate(all_masks, axis=0)
            y_test = np.concatenate(all_labels, axis=0)
    elif model.model_name == "TwoTowerMLP":
        with torch.no_grad():
            miRNA_seqs = []
            mRNA_seqs = []
            miRNA_masks = []
            mRNA_masks = []
            X_test = []
            y_test = []
            for mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target in test_loader:
                miRNA_seqs.append(miRNA_seq.cpu().numpy())
                mRNA_seqs.append(mRNA_seq.cpu().numpy())
                mRNA_masks.append(mRNA_seq_mask.cpu().numpy())
                miRNA_masks.append(miRNA_seq_mask.cpu().numpy())
                seq = mRNA_seq + [4,4,4,4,4,4] + miRNA_seq
                X_test.append(seq.cpu().numpy())
                y_test.append(target.cpu().numpy())
            miRNA_seqs = np.concatenate(miRNA_seqs, axis=0)
            mRNA_seqs = np.concatenate(mRNA_seqs, axis=0)
            mRNA_masks = np.concatenate(mRNA_masks, axis=0)
            miRNA_masks = np.concatenate(miRNA_masks, axis=0)

    # 2) Evaluate model predictions
    if model.model_name == "HyenaDNA":
        X_test_torch = torch.tensor(X_test, dtype=torch.long, device=device)
        X_test_mask = torch.tensor(X_test_mask, dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(seq=X_test_torch, 
                            seq_mask=X_test_mask
                            )  # shape: (N,1)
    elif model.model_name == "TwoTowerMLP":
        miRNA_seqs = torch.tensor(miRNA_seqs, dtype=torch.long, device=device)
        mRNA_seqs = torch.tensor(mRNA_seqs, dtype=torch.long, device=device)
        mRNA_masks = torch.tensor(mRNA_masks, dtype=torch.long, device=device)
        miRNA_masks = torch.tensor(miRNA_masks, dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(mRNA_seq=mRNA_seqs,
                            miRNA_seq=miRNA_seqs,
                            mRNA_seq_mask=mRNA_masks,
                            miRNA_seq_mask=miRNA_masks
                            )
    probs = torch.sigmoid(outputs.squeeze()).cpu().numpy().flatten()
    predictions = (probs > 0.5).astype(int)
            
    # Pick the most correct samples to explain
    # Number of classes
    classes = np.unique(y_test)
    print("Number of classes = ", classes)

    # Initialize a dictionary to store selected indices
    best_index_for_class = {}
    confidences = np.where(predictions==1, probs, 1 - probs)  # if pred=1: prob, else 1-prob
    for cls in classes:
        mask = (y_test == cls) & (predictions == cls)  # correct predictions of class=cls
        if np.any(mask):
            idx_candidates = np.where(mask)[0]
            cand_confs = confidences[idx_candidates]
            # Pick the highest
            best_idx = idx_candidates[np.argmax(cand_confs)]
            best_index_for_class[cls] = best_idx
        else:
            best_index_for_class[cls] = None

    # 4) Define a model_forward function for SHAP
    #    This function must take a numpy array of shape (batch, seq_len) integer IDs
    #    and return probabilities or outputs in numpy
    def Hyena_forward(batch_of_tokens):
        # shape: (B, seq_len)
        t = torch.tensor(batch_of_tokens, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(t)  # (B, n_classes) or (B,1)
            if out.shape[1] == 1:
                # binary
                out = torch.sigmoid(out).cpu().numpy()  # shape: (B,1)
            else:
                out = nn.functional.softmax(out, dim=1).cpu().numpy()  # shape: (B, n_classes)
        return out
    
    def TwoTower_forward(mRNA_tokens, miRNA_tokens):
        return

    # Create background
    sample_tokens = X_test[np.random.choice(X_test.shape[0], size=1)] # (1, seq_len)
    indices = np.random.choice(X_test.shape[0], size=min(n_background, X_test.shape[0]), replace=False)
    background_data = X_test[indices, :]  # (B, seq_len)
    # Convert the background using background_function:
    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],  # add RNA characters, N is uncertain
        model_max_length=model.mRNA_max_len + model.miRNA_max_len,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )
    background_processed = background_function(background_data, tokenizer)
    background_torch = torch.tensor(background_processed, dtype=torch.long, device=device)
    background_mask_torch = torch.full(background_torch.size(), 1, dtype=torch.long, device=device)
    
    print("Background shape = ", background_torch.shape)

    # 5) Create the SHAP explainer
    #    We can choose shap.DeepExplainer, shap.GradientExplainer, shap.KernelExplainer, etc.
    #    For a PyTorch model, shap.DeepExplainer might require the actual model object, but it typically
    #    expects the model layers as a graph. Alternatively, shap.GradientExplainer can often handle
    #    embedding-based models. We'll do shap.Explainer with a custom forward function:
    # if model.model_name == "HyenaDNA":
    #     explainer = shap.DeepExplainer(Hyena_forward, background_torch)
    # elif model.model_name == "TwoTowerMLP":
    #     explainer = shap.DeepExplainer(TwoTower_forward, background_torch)
    
    # model is full-ly differentiable
    # model_forward wrapper
    # force mask to be part of the graph
    def model_forward(tokens, mask):
        # Ensure tokens are in the correct integer type
        tokens = tokens.long()
        # Convert mask to float so that it can participate in gradient computations.
        mask = mask.float()
        # Add a dummy differentiable operation using the mask
        dummy = torch.sum(mask) * 0.0
        # Call your original model; ensure it uses both tokens and mask.
        output = model(tokens, mask)
        return output + dummy  # dummy forces mask into the computation graph
    
    explainer = shap.GradientExplainer(model_forward, 
                                       [background_torch, background_mask_torch])
    
    # 8) For each class 0,1, compute shap for the best index if it exists
    fig, axes = plt.subplots(1, len(classes), figsize=(10,4))
    for i, cls in enumerate(classes):
        ax = axes[i] if len(classes) > 1 else axes
        best_idx = best_index_for_class[cls]
        if best_idx is None:
            ax.axis("off")
            ax.set_title(f"No correct example found for class {cls}")
            continue
        
        # Single sample
        sample_tokens = X_test[best_idx]  # shape (seq_len)
        sample_mask = X_test_mask[best_idx] # shape (seq_len)
        sample_torch = torch.tensor(sample_tokens, dtype=torch.long, device=device).unsqueeze(0)
        sample_mask_torch = torch.tensor(sample_mask, dtype=torch.long, device=device).unsqueeze(0)
        print("Best sample shape = ", sample_tokens.shape)
        shap_values = explainer.shap_values([sample_torch, sample_mask_torch]) # shap.Explanation with shape (1, seq_len)
        # shap_values.values => shape (1, seq_len)

        # 9) shap_values[0] => an array of length seq_len
        # for the effect on model's prob for class=1
        # if shap_values has separate "base values", etc. we can do:
        if isinstance(shap_values, list):
            # Multi-class: pick the shap values for the predicted class
            predicted_label = predictions[best_idx]
            sv_array = shap_values[predicted_label][0]  # shape: (seq_len,)
        else:
            sv_array = shap_values[0]  # shape: (seq_len,)

        # 10) Project onto the actual base to produce shape (4, seq_len)
        projected = tokens_to_basewise_shap(sample_tokens, sv_array)

        # 11) Plot with viz_sequence.plot_weights, which expects shape (seq_len,4)
        #     A,C,G,T in columns typically. We used row 0=A,1=C,2=G,3=T => transpose
        # Note: the typical convention for plot_weights is also that index 0 is A, 1=C, 2=G, 3=T
        # So we are consistent.
        viz_sequence.plot_weights(projected.T, subticks_frequency=10, ax=ax)
        conf = probs[best_idx] if cls == 1 else (1 - probs[best_idx])
        ax.set_title(f"Class={cls}, correct idx={best_idx}, conf={conf:.3f}")

    plt.tight_layout()
    # save figure
    test_dataset_name = os.path.basename(model.test_dataset_path).split('.')[0]
    save_path = os.path.join(PROJ_HOME, f"{model.model_name}_{test_dataset_name}_projected_shap.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved SHAP sequence logo figure to {save_path}")

def deeplift_analysis(model):
    """
    Implemented DeepLIFT via shap.DeepExplainer to visualize the test sequences that the model
    used for the classification. Provides visualizations for the most confident correct and
    incorrect predictions for each class.
    """

    model.seed_everything(seed=42)

    ckpt_path = os.path.join(PROJ_HOME, 
                            "checkpoints", 
                            model.dataset_name, 
                            model.model_name, 
                            str(model.mRNA_max_len), 
                            "checkpoint_epoch_final.pth")
    loaded_data = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(loaded_data["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],  # add RNA characters, N is uncertain
        model_max_length=model.max_length,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )

    D_test = model.load_dataset(model.test_dataset_path)
    if model.base_model_name == 'HyenaDNA':
        ds_test = miRawDataset(
            D_test,
            mRNA_max_length=model.mRNA_max_len, # pad to mRNA max length
            miRNA_max_length=model.miRNA_max_len, # pad to miRNA max length
            tokenizer=tokenizer,
            use_padding=model.use_padding,
            rc_aug=model.rc_aug,
            add_eos=model.add_eos,
            concat=True,
        )
    elif model.base_model_name == 'TwoTowerMLP':
        ds_test = miRawDataset(
            D_test,
            mRNA_max_length=model.mRNA_max_len, # pad to mRNA max length
            miRNA_max_length=model.miRNA_max_len, # pad to miRNA max length
            tokenizer=tokenizer,
            use_padding=model.use_padding,
            rc_aug=model.rc_aug,
            add_eos=model.add_eos,
            concat=False,
        )
    test_loader = DataLoader(ds_test, batch_size=model.batch_size, shuffle=False)

    evaluate_and_explain(model=model,
                        test_loader=test_loader,
                         device=model.device,
                         )

def main():
    parser = argparse.ArgumentParser(description="Run Deeplift analysis on test mRNA sequences")
    parser.add_argument(
        "--mRNA_max_length",
        type=int,
        default=1000,
        help="Maximum length of mRNA sequences (default: 1000)",
    )
    parser.add_argument(
        "--miRNA_max_length",
        type=int,
        default=28,
        help="Maximum length of mRNA sequences (default: 28)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run training on (default: auto-detected)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size to load training dataset"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="Model to train/test"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Name of the folder where model checkpoints, model train loss and test accuracies are saved."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="train dataset",
        help="The name of the folder that indicate which training dataset is used"
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="path/to/train/dataset",
        help="Path to training dataset"
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default="path/to/validation/dataset",
        help="Path to validation dataset"
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="path/to/testdataset",
        required=False,
        help="Path to test dataset"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model on test dataset"
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use DistributedDataParallel for multi-GPU training."
    )
    parser.add_argument(
        "--resume_ckpt", 
        type=str, 
        default=None, 
        help="Path to checkpoint to resume from."
    )
    parser.add_argument(
        "--backbone_cfg",
        default=None,
        required=False
    )
    parser.add_argument(
        "--basemodel_cfg",
        type=str,
        required=False,
    )
    # exps params
    parser.add_argument(
        "--no_use_padding",
        action="store_false",
        dest="use_padding",
        required=False,
        help="Disabling padding (default is to use padding)"
    )
    parser.add_argument(
        "--rc_aug",
        action="store_true",
        help="Enable reverse complement augmentation (default: False)"
    )
    parser.add_argument(
        "--add_eos",
        action="store_true",
        help="Enable adding end-of-sentence token (default: False)"
    )
    parser.add_argument(
        "--use_head",
        action="store_true",
        help="Enable use of decoder head (default: False)"
    )
    parser.add_argument(
        "--accumulation_step",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=1,
        required=False,
    )
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # load checkpoint of models
    model = load_model(**arg_dict)
    
    # perform DeepLIFT analysis
    deeplift_analysis(model)
    
if __name__ == '__main__':
    main()