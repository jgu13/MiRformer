import os
import optuna
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from Data_pipeline import miRawDataset, CharacterTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from argument_parser import get_argument_parser

# Example: Assume these are your two model classes
# from HyenaDNAWrapper import HyenaDNAWrapper
# from TwoTowerMLP import TwoTowerMLP

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

def objective(trial, 
              **kwargs):
    """
    The Optuna objective function.

    Args:
        trial (optuna.trial.Trial): The trial object for hyperparameter suggestions.
        model_name (str): Either 'HyenaDNA' or 'TwoTowerMLP' or something else you use in your factory logic.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        device (str): The device to run on (e.g., 'cuda' or 'cpu').

    Returns:
        float: The validation accuracy after training with the suggested hyperparameters.
    """
    model_name = kwargs.get("model_name")
    
    # ==========================
    # 1. Suggest Hyperparameters
    # ==========================
    learning_rate = trial.suggest_categorical("lr", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    accumulation_step = trial.suggest_categorical("accumulation_step", [2, 4, 8])
    alpha = trial.suggest_categorical("alpha", [0.01, 0.05, 0.1, 0.2, 0.5])
    margin = trial.suggest_categorical("margin", [0.01, 0.05, 0.1, 0.2, 0.3])
    
    # If model is TwoTowerMLP, we might tune hidden_sizes
    if model_name == "TwoTowerMLP":
        hidden_sizes = []
        num_layers = trial.suggest_int("num_hidden_layers", 1, 3)
        hsize = trial.suggest_int(f"hidden_size", 128, 512, step=128)
        for _ in range(num_layers):
            hidden_sizes.append(hsize)
    else:
        hidden_sizes = None  # Not used for HyenaDNA

    # Number of epochs (could also be a hyperparameter, or fixed)
    epochs = kwargs.get("epochs")
    
    print(f"Trial:{trial.number}\n"
          f"Learning rate = {learning_rate}\n"
          f"weight decay = {weight_decay}\n"
          f"accumulation step = {accumulation_step}\n"
          f"Hidden sizes = {hidden_sizes}\n"
          f"Alpha = {alpha}\n"
          f"Margin = {margin}\n"
          f"Epochs = {epochs}")
    
    # ===================
    # 2. Instantiate the Model
    # ===================
    # Example logic: we pass in the hyperparams to the model. 
    # Adjust to match your own constructor signature.
    base_model_name = kwargs.get("base_model_name")
    if base_model_name == "HyenaDNA":
        from HyenaDNAWrapper import HyenaDNAWrapper
        model = HyenaDNAWrapper(
            **kwargs,
        )
    elif base_model_name == "TwoTowerMLP":
        from TwoTowerMLP import TwoTowerMLP
        model = TwoTowerMLP(
            hidden_sizes=hidden_sizes,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.accumulation_step = accumulation_step

    # The model might internally store lr, weight_decay, etc.
    # If not, we handle them externally in the optimizer creation below.

    # Transfer model to device
    model.to(model.device)
    
    # ===============
    # 2. Load Dataset
    # ===============
    # create tokenizer
    print("Load datasets...")
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],  # add RNA characters, N is uncertain
        model_max_length=model.max_length,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="left",  # since HyenaDNA is causal, we pad on the left
    )
    D_train = model.load_dataset(model.train_dataset_path)
    D_val = model.load_dataset(model.val_dataset_path)
    
    if model.base_model_name == 'HyenaDNA':
        ds_train = miRawDataset(
            D_train,
            mRNA_max_length=model.mRNA_max_len,
            miRNA_max_length=model.miRNA_max_len,
            seed_start_col="seed start",
            seed_end_col="seed end",
            tokenizer=tokenizer,
            use_padding=model.use_padding,
            rc_aug=model.rc_aug,
            add_eos=model.add_eos,
            concat=True,
            add_linker=True,
        )
        ds_val = miRawDataset(
            D_val,
            mRNA_max_length=model.mRNA_max_len,
            miRNA_max_length=model.miRNA_max_len,
            seed_start_col="seed start",
            seed_end_col="seed end",
            tokenizer=tokenizer,
            use_padding=model.use_padding,
            rc_aug=model.rc_aug,
            add_eos=model.add_eos,
            concat=True,
            add_linker=True,
        )
    elif model.base_model_name == 'TwoTowerMLP':
        ds_train = miRawDataset(
            D_train,
            mRNA_max_length=model.mRNA_max_len,
            miRNA_max_length=model.miRNA_max_len,
            seed_start_col="seed start",
            seed_end_col="seed end",
            tokenizer=tokenizer,
            use_padding=model.use_padding,
            rc_aug=model.rc_aug,
            add_eos=model.add_eos,
            concat=False,
        )
        ds_val = miRawDataset(
            D_val,
            mRNA_max_length=model.mRNA_max_len,
            miRNA_max_length=model.miRNA_max_len,
            seed_start_col="seed start",
            seed_end_col="seed end",
            tokenizer=tokenizer,
            use_padding=model.use_padding,
            rc_aug=model.rc_aug,
            add_eos=model.add_eos,
            concat=False,
        )
    
    train_loader = DataLoader(ds_train, 
                            batch_size=model.batch_size,  
                            shuffle=True)
    test_loader = DataLoader(ds_val, 
                            batch_size=model.batch_size, 
                            shuffle=False) 

    # ===================
    # 3. Create Optimizer
    # ===================
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()  # or cross-entropy, etc.

    # ===================
    # 4. Training Loop
    # ===================
    print("Starts training!")
    best_acc = 0
    checkpoint_dir = os.path.join(
            PROJ_HOME, 
            "checkpoints", 
            model.dataset_name, 
            model.model_name, 
            str(model.mRNA_max_len),
        )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        train_loss = model.run_training(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_fn=loss_fn,
            tokenizer=tokenizer,
            margin=margin,
            alpha=alpha,
        )
        # Evaluate or skip
        test_acc = model.run_testing(
            model=model,
            test_loader=test_loader,
        )

        # Save the checkpoint if validation accuracy improves
        if test_acc > best_acc:
            best_val_acc = test_acc
            best_train_loss = train_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint_trial_{trial.number}_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            best_checkpoint_path = checkpoint_path
        
        # Report these intermediate metrics to Optuna
        # so they appear in the "Intermediate values" chart
        trial.report(train_loss, step=epoch)
        trial.report(test_acc, step=epoch + 0.5) # offset by 0.5 so we can see it
        trial.set_user_attr(model_name, f"trial_{trial.number}")
        
        # implement early stopping
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Save best results in trial's user attributes
    trial.set_user_attr("best_checkpoint", best_checkpoint_path)
    trial.set_user_attr("best_train_loss", best_train_loss)
    trial.set_user_attr("best_val_acc", best_val_acc)
    trial.set_user_attr("best_epoch", best_epoch)

    # Return the metric that Optuna will try to maximize or minimize
    return test_acc

def main():
    # Parse command-line arguments
    parser = get_argument_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    model_name = args_dict.get("model_name")
    import os
    database_path = os.path.join(os.path.expanduser("~/projects/mirLM"), "optuna_database", f"{model_name}_study.db")
    
    # Define an Optuna study
    study = optuna.create_study(
        study_name="mirLM_study",
        storage=f"sqlite://///{database_path}",
        direction="maximize",
        load_if_exists=True,
    )  # for maximizing accuracy
    print("------- Optimization starts here ------")
    # Optimize
    study.optimize(
        lambda trial: objective(
            trial,
            **args_dict,
        ),
        n_trials=40,  # number of trials
    )

    # 5) Print best result
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    main()