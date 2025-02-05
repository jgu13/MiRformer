import os
import json
import time
import math
import torch
import argparse
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader

# Local imports
from TwoTowerMLP import TwoTowerMLP
from Data_pipeline import CharacterTokenizer, miRawDataset

PROJ_HOME = os.path.expanduser("~/projects/mirLM")


def train(
    TwoTowerMLP,
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
    TwoTowerMLP.train()
    
    loss_ls = []
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
        output = TwoTowerMLP(
            device=device,
            mRNA_seq=mRNA_seq,
            miRNA_seq=miRNA_seq,
            mRNA_seq_mask=mRNA_seq_mask,
            miRNA_seq_mask=miRNA_seq_mask,
        )
        loss = loss_fn(output.squeeze().sigmoid(), target.squeeze())
        # print("Output = ", output[0:10])
        # accumulation_step = 1
        if accumulation_step is not None:
            loss = loss / accumulation_step
            loss_ls.append(loss.item())
            loss.backward()
            if (batch_idx + 1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(
                    f"Train Epoch: {epoch} "
                    f"[{(batch_idx + 1) * len(mRNA_seq)}/{len(train_loader.dataset)} "
                    f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)]\t"
                    f"Avg Loss: {sum(loss_ls) / len(loss_ls):.6f}\t", 
                    flush=True
                )
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} "
                    f"[{(batch_idx + 1) * len(mRNA_seq)}/{len(train_loader.dataset)} "
                    f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}\t"
                )
        epoch_loss += loss.item()
    average_loss = epoch_loss / len(train_loader)
    return average_loss


def test(TwoTowerMLP,
         device, 
         test_loader):
    """Test loop."""
    TwoTowerMLP.train()
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
            output = TwoTowerMLP(
                device=device,
                mRNA_seq=mRNA_seq,
                miRNA_seq=miRNA_seq,
                mRNA_seq_mask=mRNA_seq_mask,
                miRNA_seq_mask=miRNA_seq_mask,
            )
            probabilities = torch.sigmoid(output.squeeze())
            predictions = (probabilities > 0.5).long()
            # print('test predictions', predictions)
            correct += predictions.eq(target.squeeze()).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({accuracy:2f})"
    )
    return accuracy


def evaluate(TwoTowerMLP,
            device, 
            test_loader,
            ):
    """Test loop."""
    TwoTowerMLP.train()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target in test_loader:
            mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target = (
                mRNA_seq.to(device),
                miRNA_seq.to(device),
                mRNA_seq_mask.to(device),
                miRNA_seq_mask.to(device),
                target.to(device),
            )
            output = TwoTowerMLP(
                device=device,
                mRNA_seq=mRNA_seq,
                miRNA_seq=miRNA_seq,
                mRNA_seq_mask=mRNA_seq_mask,
                miRNA_seq_mask=miRNA_seq_mask,
            )
            probabilities = torch.sigmoid(output.squeeze()).cpu().numpy().tolist()
            targets = target.cpu().view(-1).numpy().tolist()
            predictions.extend(probabilities)
            true_labels.extend(targets)

    return predictions, true_labels

def load_dataset(dataset, sep=','):
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


def run(
    mRNA_length, 
    miRNA_length, 
    epochs, 
    batch_size=16,
    device="cpu",
    model_name="",
    dataset_name="",
    train_dataset_path="",
    val_dataset_path="",
    test_dataset_path="",
    evaluate_flag=False,
    backbone_cfg=None,
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
    # pretrained_model_name = "hyenadna-small-32k-seqlen"  # use None if training from scratch
    pretrained_model_name = None

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 1

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one by default
    backbone_cfg = backbone_cfg
    download = False

    # instantiate the model (pretrained here)
    model = TwoTowerMLP(
        pretrained_model_name=pretrained_model_name,
        backbone_cfg=backbone_cfg,
        n_classes=n_classes,
        use_head=use_head,
        device=device,
        download=download,
    )
    
    model.to(device)
    
    if evaluate_flag:
        # load checkpoint
        ckpt_path = os.path.join(PROJ_HOME, "checkpoints", dataset_name, model_name, str(mRNA_max_length), "checkpoint_epoch_final.pth")
        loaded_data = torch.load(ckpt_path, map_location=device)
        print("Loaded checkpoint from ", ckpt_path)
        model.load_state_dict(loaded_data["model_state_dict"])

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T"],  # add RNA characters, N is uncertain
        model_max_length=mRNA_max_length
                        + miRNA_max_length
                        + 2,  # to account for miRNA length and special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )

    if evaluate_flag:
        D_test = load_dataset(test_dataset_path)
        ds_test = miRawDataset(
            D_test,
            mRNA_max_length=mRNA_max_length, # pad to mRNA max length
            miRNA_max_length=miRNA_max_length, # pad to miRNA max length
            tokenizer=tokenizer,
            use_padding=use_padding,
            rc_aug=rc_aug,
            add_eos=add_eos,
            concat=False,
        )
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    else:
        D_train = load_dataset(train_dataset_path)
        D_val = load_dataset(val_dataset_path)
        
        ds_train = miRawDataset(
            D_train,
            mRNA_max_length=mRNA_max_length,
            miRNA_max_length=miRNA_max_length,
            tokenizer=tokenizer,
            use_padding=use_padding,
            rc_aug=rc_aug,
            add_eos=add_eos,
            concat=False,
        )
        ds_test = miRawDataset(
            D_val,
            mRNA_max_length=mRNA_max_length,
            miRNA_max_length=miRNA_max_length,
            tokenizer=tokenizer,
            use_padding=use_padding,
            rc_aug=rc_aug,
            add_eos=add_eos,
            concat=False,
        )

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

        # loss function
        loss_fn = nn.BCELoss()

        # create optimizer
        optimizer = optim.AdamW(
            model.parameters()
        )

    if evaluate_flag:
        test_dataset_name = os.path.basename(test_dataset_path).split('.')[0]
        predictions, true_labels = evaluate(
            model,
            device=device,
            test_loader=test_loader,
        )
        # Save predictions and true labels
        output_path = os.path.join(PROJ_HOME, "Performance", test_dataset_name, model_name)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f"predictions_{mRNA_max_length}.json"), "w") as f:
            json.dump({"predictions": predictions, "true_labels": true_labels}, f)

        print("Evaluation completed. Predictions saved to", output_path)
    else:
        average_loss_list = []
        accuracy_list = []

        model_checkpoints_dir = os.path.join(
            PROJ_HOME, "checkpoints", dataset_name, model_name, f"{mRNA_length}"
        )
        os.makedirs(model_checkpoints_dir, exist_ok=True)

        start = time.time()
        
        print("Binary Classification -- Start training --")
        print("mRNA length:", mRNA_max_length)
        print(f"For {epochs} epochs")
        print("Using device:", device)
        
        for epoch in range(epochs):
            average_loss = train(
                model,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                loss_fn=loss_fn,
                accumulation_step=accumulation_step,
            )
            accuracy = test(
                model,            
                device=device,
                test_loader=test_loader,
            )
            average_loss_list.append(average_loss)
            accuracy_list.append(accuracy)
            
            # save checkpoints
            if (epoch + 1) % 10 == 0:
                # Save the model checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": average_loss,
                        "accuracy": accuracy,
                    },
                    os.path.join(model_checkpoints_dir, f"checkpoint_epoch_{epoch}.pth"),
                )

        # save the final model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss,
                "accuracy": accuracy,
            },
            os.path.join(model_checkpoints_dir, f"checkpoint_epoch_final.pth"),
        )
        time_taken = time.time() - start
        print(f"Time taken for {epochs} epochs = {(time_taken / 60):.2f} min.")

        # Save test_accuracy and train_average_loss
        perf_dir = os.path.join(PROJ_HOME, "Performance", dataset_name, model_name)
        os.makedirs(perf_dir, exist_ok=True)

        with open(
            os.path.join(perf_dir, f"test_accuracy_{mRNA_max_length}.json"),
            "w",
            encoding="utf-8"
        ) as fp:
            json.dump(accuracy_list, fp)

        with open(
            os.path.join(perf_dir, f"train_loss_{mRNA_max_length}.json"),
            "w",
            encoding="utf-8"
        ) as fp:
            json.dump(average_loss_list, fp)
        return


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
        "--miRNA_max_len",
        type=int,
        default=28,
        help="Maximum length of mRNA sequences (default: 28)",
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
        "--model_name",
        type=str,
        required=True,
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
        help="Path to test dataset"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model on test dataset"
    )
    parser.add_argument(
        "--backbone_cfg",
        default=None,
        required=False
    )
    args = parser.parse_args()

    # Extract arguments
    mRNA_max_length = args.mRNA_max_len
    miRNA_max_length = args.miRNA_max_len
    device = torch.device(args.device) if torch.cuda.is_available() else "cpu"
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    model_name = args.model_name
    dataset_name = args.dataset_name
    train_dataset_path = args.train_dataset_path
    val_dataset_path = args.val_dataset_path
    test_dataset_path = args.test_dataset_path
    evaluate_flag = args.evaluate
    backbone_cfg=args.backbone_cfg


    # Launch training or evaluation
    run(
        mRNA_length=mRNA_max_length,
        miRNA_length=miRNA_max_length,
        epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        model_name=model_name,
        dataset_name=dataset_name,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        test_dataset_path=test_dataset_path,
        evaluate_flag=evaluate_flag,
        backbone_cfg=backbone_cfg,
    )

if __name__ == "__main__":
    main()

