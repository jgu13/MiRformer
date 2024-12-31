import os
import json
import time
import math
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from Hyena_layer import HyenaDNAPreTrainedModel, HyenaDNAModel, LinearHead
from Data_pipeline import CharacterTokenizer, CustomDataset

PROJ_HOME = os.path.expanduser("~/projects/mirLM")


def compute_cross_attention(Q: torch.tensor, 
                            K: torch.tensor, 
                            V: torch.tensor, 
                            Q_mask: torch.tensor, 
                            K_mask: torch.tensor):
    '''
    Compute cross attention
    '''
    d_model = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
        d_model
    )  # [batchsize, mRNA_seq_len, miRNA_seq_len]
    # expand K mask to mask out keys, make each query only attend to valid keys
    K_mask = K_mask.unsqueeze(1).expand(
        -1, Q.shape[1], -1
    )  # [batchsize, mRNA_seq_len, miRNA_seq_len]
    scores = scores.masked_fill(K_mask == 0, -1e9)
    # apply softmax on the key dimension
    attn_weights = F.softmax(scores, dim=-1)  # [batchsize, mRNA_seq_len, miRNA_seq_len]
    cross_attn = torch.matmul(attn_weights, V)  # [batchsize, mRNA_seq_len, d_model]
    # expand Q mask to mask out queries, zero out padded queries
    valid_counts = Q_mask.sum(dim=1, keepdim=True)  # [batchsize, 1]
    Q_mask = Q_mask.unsqueeze(-1).expand(
        -1, -1, d_model
    )  # [batchsize, mRNA_seq_len, d_model]
    cross_attn = cross_attn * Q_mask
    # average pool over seq_length
    cross_attn = cross_attn.sum(dim=1) / valid_counts  # [batchsize, d_model]
    # print("Cross attention shape = ", cross_attn.shape)
    return cross_attn


def train(
    HyenaDNA_feature_extractor,
    MLP_head,
    Q_layer,
    KV_layer,
    device,
    train_loader,
    optimizer,
    epoch,
    loss_fn,
    log_interval=10,
    epoch_loss=0.0,
    accumulation_step=None,
):
    """Training loop."""
    MLP_head.eval()
    HyenaDNA_feature_extractor.eval()
    for batch_idx, (
        mRNA_seq,
        miRNA_seq,
        mRNA_seq_mask,
        miRNA_seq_mask,
        target,
    ) in enumerate(train_loader):
        mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target = (
            mRNA_seq.to(device),
            miRNA_seq.to(device),
            mRNA_seq_mask.to(device),
            miRNA_seq_mask.to(device),
            target.to(device),
        )
        # torch.save({"mRNA_seq": mRNA_seq,
        #             "miRNA_seq": miRNA_seq,
        #             "mRNA_seq_mask": mRNA_seq_mask,
        #             "miRNA_seq_mask": miRNA_seq_mask,
        #             "target": target},
        #            os.path.join(PROJ_HOME, "data", "MLP_test_batch.pth"))
        # exit()
        # batch_data = torch.load("/home/mcb/users/jgu13/projects/mirLM/data/MLP_test_batch.pth")
        # mRNA_seq = batch_data["mRNA_seq"]
        # miRNA_seq = batch_data["miRNA_seq"]
        # mRNA_seq_mask = batch_data["mRNA_seq_mask"]
        # miRNA_seq_mask = batch_data["miRNA_seq_mask"]
        # target = batch_data["target"]
        with torch.no_grad():
            mRNA_hidden_states = HyenaDNA_feature_extractor(
                device=device, input_ids=mRNA_seq, use_only_miRNA=False
            )  # (batchsize, mRNA_seq_len, hidden), mask size = (batchsize, mRNA_seq_len)
            miRNA_hidden_states = HyenaDNA_feature_extractor(
                device=device, input_ids=miRNA_seq, use_only_miRNA=False
            )  # (batchsize, miRNA_seq_len, hidden), mask size = (batchsize, miRNA_seq_len)
        # cross attn between mRNA and miRNA embeddings
        d_model = mRNA_hidden_states.shape[-1]
        Q = Q_layer(miRNA_hidden_states)  # [batchsize, miRNA_seq_len, d_model]
        K = KV_layer(mRNA_hidden_states)  # [batchsize, mRNA_seq_len, d_model]
        # V = KV_layer(mRNA_hidden_states)  # [batchsize, mRNA_seq_len, d_model]
        # print("Q information = ", Q[0:2, 0:2, 0:10])
        # print("K information = ", K[0:2, 0:2, 0:10])
        # Q_mask = miRNA_seq_mask
        Q_mask = miRNA_seq_mask.unsqueeze(-1).expand(
            -1, -1, d_model
        )
        K_mask = mRNA_seq_mask.unsqueeze(-1).expand(
            -1, -1, d_model
        )
        # K_mask = mRNA_seq_mask
        Q_valid_counts = miRNA_seq_mask.sum(dim=1, keepdim=True) # [batchsize, 1]
        K_valid_counts = mRNA_seq_mask.sum(dim=1, keepdim=True) # [batchsize, 1]
        cross_attn = torch.sum(Q * Q_mask, dim=1) / Q_valid_counts + \
                    torch.sum(K * K_mask, dim=1) / K_valid_counts
        # print("cross attn information = ", cross_attn[0:5, 0:10])
        # cross_attn = compute_cross_attention(
        #     Q=Q, 
        #     K=K, 
        #     V=V, 
        #     Q_mask=Q_mask, 
        #     K_mask=K_mask
        # )
        output = MLP_head(cross_attn)  # (batch_size, 1)
        loss = loss_fn(output.squeeze().sigmoid(), target.squeeze())
        # print("Output = ", output[0:10])
        accumulation_step = 1
        if accumulation_step is not None:
            loss = loss / accumulation_step
            loss.backward()
            if batch_idx % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(
                    f"Train Epoch: {epoch} "
                    f"[{batch_idx * len(mRNA_seq)}/{len(train_loader.dataset)} "
                    f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}\t", 
                    flush=True
                )
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} "
                    f"[{batch_idx * len(mRNA_seq)}/{len(train_loader.dataset)} "
                    f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}\t"
                )
        epoch_loss += loss.item()
    average_loss = epoch_loss / len(train_loader)
    return average_loss


def test(HyenaDNA_feature_extractor, 
         MLP_head, 
         Q_layer,
         KV_layer,
         device, 
         test_loader):
    """Test loop."""
    MLP_head.eval()
    correct = 0
    with torch.no_grad():
        for mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target in test_loader:
            mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target = (
                mRNA_seq.to(device),
                miRNA_seq.to(device),
                mRNA_seq_mask.to(device),
                miRNA_seq_mask.to(device),
                target.to(device),
            )
            
            # batch_data = torch.load("/home/mcb/users/jgu13/projects/mirLM/data/MLP_test_batch.pth")
            # mRNA_seq = batch_data["mRNA_seq"]
            # miRNA_seq = batch_data["miRNA_seq"]
            # mRNA_seq_mask = batch_data["mRNA_seq_mask"]
            # miRNA_seq_mask = batch_data["miRNA_seq_mask"]
            # target = batch_data["target"]
            mRNA_hidden_states = HyenaDNA_feature_extractor(
                device=device, input_ids=mRNA_seq, use_only_miRNA=False
            )
            miRNA_hidden_states = HyenaDNA_feature_extractor(
                device=device, input_ids=miRNA_seq, use_only_miRNA=False
            )
            d_model = mRNA_hidden_states.shape[-1]
            Q = Q_layer(miRNA_hidden_states)  # [batchsize, miRNA_seq_len, d_model]
            K = KV_layer(mRNA_hidden_states)  # [batchsize, mRNA_seq_len, d_model]
            # V = KV_layer(mRNA_hidden_states)  # [batchsize, mRNA_seq_len, d_model]
            # Q_mask = miRNA_seq_mask
            Q_mask = miRNA_seq_mask.unsqueeze(-1).expand(
                -1, -1, d_model
            )
            K_mask = mRNA_seq_mask.unsqueeze(-1).expand(
                -1, -1, d_model
            )
            # K_mask = mRNA_seq_mask
            Q_valid_counts = miRNA_seq_mask.sum(dim=1, keepdim=True) # [batchsize, 1]
            K_valid_counts = mRNA_seq_mask.sum(dim=1, keepdim=True) # [batchsize, 1]
            cross_attn = torch.sum(Q * Q_mask, dim=1) / Q_valid_counts + \
                        torch.sum(K * K_mask, dim=1) / K_valid_counts
            output = MLP_head(cross_attn)  # (batch_size, 1)
            probabilities = torch.sigmoid(output.squeeze())
            predictions = (probabilities > 0.5).long()
            print('test predictions', predictions)
            correct += predictions.eq(target.squeeze()).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({accuracy:2f})"
    )
    return accuracy


def load_external_dataset(dataset, sep=','):
    """
    `dataset` can be a path to the 
    locally saved dataset or it can be an 
    loaded pandas DataFrame
    """
    if dataset:
        if isinstance(dataset, str):
            if dataset.endswith((".csv", ".txt", ".tsv")):
                data = pd.read_csv(dataset, sep=sep)
            elif dataset.endswith(".xlsx"):
                data = pd.read_excel(dataset)
            elif dataset.endswith(".json"):
                with open(dataset, "r", encoding="utf-8") as f:
                    data = pd.read_json(f)
            else:
                raise ValueError(f"Unrecognized format of {dataset}")
        elif isinstance(dataset, pd.DataFrame):
            data = dataset
        else:
            raise TypeError("Dataset must be a path or a pandas DataFrame.")
    else:
        raise ValueError("Dataset cannot be None.")
    return data


def run_train(
    mRNA_length, 
    miRNA_length, 
    epochs, 
    batch_size=16,
    external_dataset=None, 
    device="cpu",
    dataset_name="",
):
    """
    Main entry point for training.
    """
    # experiment settings:
    mRNA_max_length = mRNA_length  # max len of sequence of dataset (of what you want)
    miRNA_max_length = miRNA_length
    use_padding = True
    accumulation_step = 256 // batch_size
    rc_aug = False  # reverse complement augmentation
    add_eos = True  # add end of sentence token

    # for fine-tuning, only the 'tiny' model can fit on colab
    pretrained_model_name = "hyenadna-small-32k-seqlen"  # use None if training from scratch

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 1

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one by default
    backbone_cfg = None
    download = False

    # instantiate the model (pretrained here)
    if pretrained_model_name in [
        "hyenadna-tiny-1k-seqlen",
        "hyenadna-small-32k-seqlen",
        "hyenadna-medium-160k-seqlen",
        "hyenadna-medium-450k-seqlen",
        "hyenadna-large-1m-seqlen",
    ]:
        path = f"{PROJ_HOME}/checkpoints"
        # use the pretrained Huggingface wrapper instead
        HyenaDNA_feature_extractor = HyenaDNAPreTrainedModel.from_pretrained(
            path,
            pretrained_model_name,
            download=download,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
        )
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, pretrained_model_name)
        if os.path.isdir(pretrained_model_name_or_path):
            if backbone_cfg is None:
                with open(
                    os.path.join(pretrained_model_name_or_path, "config.json"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    backbone_cfg = json.load(f)
            elif isinstance(backbone_cfg, str) and backbone_cfg.endswith(".json"):
                with open(backbone_cfg, "r", encoding="utf-8") as f:
                    backbone_cfg = json.load(f)
            else:
                assert isinstance(
                    backbone_cfg, dict
                ), "self-defined backbone config must be a dictionary."
    # from scratch
    else:
        try:
            HyenaDNA_feature_extractor = HyenaDNAModel(
                **backbone_cfg, use_head=use_head, n_classes=n_classes
            )
        except TypeError as exc:
            raise TypeError("backbone_cfg must not be NoneType.") from exc

    # CNN feature extractor
    cnn_feature_extractor = nn.Conv1d()
    
    learning_rate = backbone_cfg["lr"]
    weight_decay = backbone_cfg["weight_decay"]
    hidden_sizes = []#[backbone_cfg["d_model"] * 2, backbone_cfg["d_model"] * 2]
    MLP_head = LinearHead(
        d_model=backbone_cfg["d_model"], d_output=n_classes, hidden_sizes=hidden_sizes
    )
    Q_layer = nn.Linear(backbone_cfg["d_model"], backbone_cfg["d_model"])
    KV_layer = nn.Linear(backbone_cfg["d_model"], backbone_cfg["d_model"])

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],  # add RNA characters, N is uncertain
        model_max_length=mRNA_max_length
                        + miRNA_max_length
                        + 2,  # to account for miRNA length and special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )

    D = load_external_dataset(dataset=external_dataset, 
                              sep=',')

    ds_train, ds_test = train_test_split(
        D, test_size=0.2, random_state=34, shuffle=True
    )
    ds_train = CustomDataset(
        ds_train,
        mRNA_max_length=mRNA_max_length,
        miRNA_max_length=miRNA_max_length,
        tokenizer=tokenizer,
        use_padding=use_padding,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )
    ds_test = CustomDataset(
        ds_test,
        mRNA_max_length=mRNA_max_length,
        miRNA_max_length=miRNA_max_length,
        tokenizer=tokenizer,
        use_padding=use_padding,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # loss function
    loss_fn = nn.BCELoss()

    # create optimizer
    # only optimize MLP layers
    optimizer = optim.AdamW(
        list(MLP_head.parameters()) + list(Q_layer.parameters()) + list(KV_layer.parameters())
    )

    HyenaDNA_feature_extractor.to(device)
    MLP_head.to(device)
    Q_layer.to(device)
    KV_layer.to(device)
    print(MLP_head)
    print("Leanring rate = ", learning_rate)
    print("weight decay = ", weight_decay)

    average_loss_list = []
    accuracy_list = []

    model_checkpoints_dir = os.path.join(
        PROJ_HOME, "checkpoints", dataset_name, f"mirLM-{mRNA_length}"
    )
    os.makedirs(model_checkpoints_dir, exist_ok=True)

    for epoch in range(epochs):
        average_loss = train(
            HyenaDNA_feature_extractor=HyenaDNA_feature_extractor,
            MLP_head=MLP_head,
            Q_layer=Q_layer,
            KV_layer=KV_layer,
            device=device,
            train_loader=test_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_fn=loss_fn,
            accumulation_step=accumulation_step,
        )
        accuracy = test(
            HyenaDNA_feature_extractor=HyenaDNA_feature_extractor,
            MLP_head=MLP_head,
            Q_layer=Q_layer,
            KV_layer=KV_layer,            
            device=device,
            test_loader=test_loader,
        )
        average_loss_list.append(average_loss)
        accuracy_list.append(accuracy)

    #     # save checkpoints
    #     if epoch % 10 == 0:
    #         # Save the model checkpoint
    #         torch.save(
    #             {
    #                 "epoch": epoch,
    #                 "model_state_dict": MLP_head.state_dict(),
    #                 "optimizer_state_dict": optimizer.state_dict(),
    #                 "loss": average_loss,
    #                 "accuracy": accuracy,
    #             },
    #             os.path.join(model_checkpoints_dir, f"checkpoint_epoch_{epoch}.pth"),
    #         )

    # # save the final model
    # torch.save(
    #     {
    #         "epoch": epoch,
    #         "model_state_dict": MLP_head.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "loss": average_loss,
    #         "accuracy": accuracy,
    #     },
    #     os.path.join(model_checkpoints_dir, f"checkpoint_epoch_final.pth"),
    # )

    return average_loss_list, accuracy_list


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run training script for binary classification.")
    parser.add_argument(
        "--mRNA_max_len",
        type=int,
        default=1000,
        help="Maximum length of mRNA sequences (default: 1000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run training on (default: auto-detected)",
    )
    parser.add_argument(
        "--num_epochs",
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
        "--dataset_name",
        type=str,
        default="train dataset",
        help="The name of the folder that indicate which training dataset is used"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/mcb/users/jgu13/projects/mirLM/data/training_1000.csv",
        help="Path to training dataset"
    )
    args = parser.parse_args()

    # Extract arguments
    mRNA_max_length = args.mRNA_max_len
    device = torch.device(args.device) if torch.cuda.is_available() else "cpu"
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path

    # Other fixed parameters
    miRNA_max_length = 28
    model_name = "MLP"

    print("Binary Classification -- Start training --")
    print("mRNA length:", mRNA_max_length)
    print(f"For {num_epochs} epochs")
    print("Using device:", device)

    # Launch training
    start = time.time()
    train_average_loss, test_accuracy = run_train(
        mRNA_length=mRNA_max_length,
        miRNA_length=miRNA_max_length,
        epochs=num_epochs,
        batch_size=batch_size,
        external_dataset=dataset_path,
        device=device,
        dataset_name=dataset_name
    )
    time_taken = time.time() - start
    print(f"Time taken for {num_epochs} epochs = {(time_taken / 60):.2f} min.")

    # Save test_accuracy and train_average_loss
    perf_dir = os.path.join(PROJ_HOME, "Performance", dataset_name, model_name)
    os.makedirs(perf_dir, exist_ok=True)

    # with open(
    #     os.path.join(perf_dir, f"test_accuracy_{mRNA_max_length}.json"),
    #     "w",
    #     encoding="utf-8"
    # ) as fp:
    #     json.dump(test_accuracy, fp)

    # with open(
    #     os.path.join(perf_dir, f"train_loss_{mRNA_max_length}.json"),
    #     "w",
    #     encoding="utf-8"
    # ) as fp:
    #     json.dump(train_average_loss, fp)

if __name__ == "__main__":
    main()

