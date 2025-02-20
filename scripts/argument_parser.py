import argparse

def get_argument_parser():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run training script for binary classification.")
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
    return parser