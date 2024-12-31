import os
import json
import time
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from Hyena_layer import HyenaDNAPreTrainedModel, HyenaDNAModel
from Data_pipeline import CharacterTokenizer, miRawDataset


PROJ_HOME = os.path.expanduser("~/projects/mirLM")


def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    loss_fn,
    mRNA_max_length,
    miRNA_max_length,
    log_interval=10,
    accumulation_step=None,
):
    """
    Training loop.
    """
    model.train()
    epoch_loss = 0.0
    for batch_idx, (seq, seq_mask, target) in enumerate(train_loader):
        seq, seq_mask, target = seq.to(device), seq_mask.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(
            device=device,
            input_ids=seq,
            input_mask=seq_mask,
            max_mRNA_length=mRNA_max_length,
            max_miRNA_length=miRNA_max_length,
        )
        loss = loss_fn(output.squeeze().sigmoid(), target.squeeze())
        if accumulation_step is not None:
            loss = loss / accumulation_step
            loss.backward()
            if batch_idx % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t".format(
                        epoch,
                        batch_idx * len(seq),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        else:
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t".format(
                        epoch,
                        batch_idx * len(seq),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        epoch_loss += loss.item()
    average_loss = epoch_loss / len(train_loader)
    return average_loss


def test(model, device, test_loader, mRNA_max_length, miRNA_max_length):
    """Test loop."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for seq, seq_mask, target in test_loader:
            seq, target, seq_mask = (
                seq.to(device),
                target.to(device),
                seq_mask.to(device),
            )
            output = model(
                device=device,
                input_ids=seq,
                input_mask=seq_mask,
                max_mRNA_length=mRNA_max_length,
                max_miRNA_length=miRNA_max_length,
            )
            probabilities = torch.sigmoid(output.squeeze())
            predictions = (probabilities > 0.5).long()
            correct += predictions.eq(target.squeeze()).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(
            correct, len(test_loader.dataset), accuracy
        )
    )
    return accuracy


def load_data(dataset=None, sep=","):
    """
    Load dataset from a file or DataFrame.
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


def run_train(mRNA_max_len, miRNA_max_len, dataset=None, epochs=10, device="cpu", multi_gpu=False):
    """
    Main entry point for training.
    """
    # experiment settings:
    max_length = mRNA_max_len + miRNA_max_len + 2 # +2 to account for special tokens, like EOS
    use_padding = True
    batch_size = 16
    rc_aug = False  # reverse complement augmentation
    add_eos = False  # add end of sentence token
    accumulation_step = 256 // batch_size  # effectively change batch size to 256

    # model to be used for training
    pretrained_model_name = "hyenadna-small-32k-seqlen"  # use None if training from scratch

    # we need these for the decoder head, if using
    use_head = True
    n_classes = 1

    # provide a backbone configuration, if None, pretrained model config.json will be loaded
    # if `pretrained_model_name` is defined. Otherwise,
    backbone_cfg = None

    print("Using device:", device)

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
        model = HyenaDNAPreTrainedModel.from_pretrained(
            path,
            pretrained_model_name,
            download=False,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )
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
            model = HyenaDNAModel(
                **backbone_cfg, use_head=use_head, n_classes=n_classes
            )
        except TypeError as exc:
            raise TypeError("backbone_cfg must not be NoneType.") from exc

    learning_rate = backbone_cfg["lr"]
    weight_decay = backbone_cfg["weight_decay"]

    print("learning rate = ", learning_rate)
    print("weight decay = ", weight_decay)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "U", "N"],  # add RNA characters, N is uncertain
        model_max_length=max_length,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )

    Dataset = load_data(dataset, sep=",")

    ds_train, ds_test = train_test_split(
        Dataset, test_size=0.2, random_state=34, shuffle=True
    )
    ds_train = miRawDataset(
        ds_train,
        mRNA_max_length=mRNA_max_len,
        miRNA_max_length=miRNA_max_len,
        tokenizer=tokenizer,
        use_padding=use_padding,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )
    ds_test = miRawDataset(
        ds_test,
        mRNA_max_length=mRNA_max_len,
        miRNA_max_length=miRNA_max_len,
        tokenizer=tokenizer,
        use_padding=use_padding,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    # loss function
    loss_fn = nn.BCELoss()

    # create optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.to(device)
    
    # MULTI-GPU WRAP (DataParallel)
    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    average_loss_list = []
    accuracy_list = []

    for epoch in range(epochs):
        average_loss = train(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_fn=loss_fn,
            mRNA_max_length=mRNA_max_len,
            miRNA_max_length=miRNA_max_len,
            accumulation_step=accumulation_step,
        )
        accuracy = test(
            model=model,
            device=device,
            test_loader=test_loader,
            mRNA_max_length=mRNA_max_len,
            miRNA_max_length=miRNA_max_len,
        )
        average_loss_list.append(average_loss)
        accuracy_list.append(accuracy)
        
        model_checkpoints_dir = os.path.join(
            PROJ_HOME, "checkpoints", "mirLM", f"mirLM-{mRNA_max_len}-finetune-Hyena-reverse"
        )
        os.makedirs(model_checkpoints_dir, exist_ok=True)

    #     # save checkpoints
    #     if epoch % 10 == 0:
    #         # Save the model checkpoint
    #         torch.save(
    #             {
    #                 "epoch": epoch,
    #                 "model_state_dict": model.state_dict(),
    #                 "optimizer_state_dict": optimizer.state_dict(),
    #                 "loss": average_loss,
    #                 "accuracy": accuracy,
    #             },
    #             os.path.join(
    #                 model_checkpoints_dir,
    #                 f"checkpoint_epoch_{epoch}.pth",
    #             ),
    #         )
    # # save the final model
    # torch.save(
    #     {
    #         "epoch": epoch,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "loss": average_loss,
    #         "accuracy": accuracy,
    #     },
    #     os.path.join(
    #         model_checkpoints_dir,
    #         "checkpoint_epoch_final.pth",
    #     ),
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
        "--multi_gpu",
        action="store_true",
        help="Use multiple GPUs if available (DataParallel).",
    )
    args = parser.parse_args()

    # Extract arguments
    mRNA_max_length = args.mRNA_max_len
    device = args.device
    num_epochs = args.num_epochs
    multi_gpu_flag = args.multi_gpu
    
    # If user wants multi-GPU, prefer 'cuda' (rather than 'cuda:0')
    if multi_gpu_flag and torch.cuda.is_available():
        device = "cuda"

    # Other fixed parameters
    miRNA_max_length = 28
    dataset_name = "mirLM"
    model_name = "HyenaDNA"

    print("Binary Classification -- Start training --")
    print("mRNA length:", mRNA_max_length)
    print(f"For {num_epochs} epochs")
    print("Using device:", device)

    # Load dummy test dataset
    PROJ_HOME = os.getcwd()  # Assuming current working directory as project home
    external_data_path = os.path.join(PROJ_HOME, "data", f"training_{mRNA_max_length}.csv")

    # Launch training
    start = time.time()
    train_average_loss, test_accuracy = run_train(
        mRNA_max_len=mRNA_max_length,
        miRNA_max_len=miRNA_max_length,
        epochs=num_epochs,
        dataset=external_data_path,
        device=device,
        multi_gpu=multi_gpu_flag
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
