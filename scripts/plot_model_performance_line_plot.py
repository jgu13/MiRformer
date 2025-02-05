import os
import json
import argparse
import matplotlib.pyplot as plt

def plot_performance(
    mRNA_max_len,
    dataset_name,
    model_dirs,
    model_names=None,
    training_loss_save_path=None,
    test_acc_save_path=None
):
    """
    Load training losses and test accuracies from list of JSON files,
    plot them in two figures
    """
    performance_dir = os.path.join(os.path.expanduser("~/projects/mirLM/Performance"), dataset_name)
    model_paths = [os.path.join(performance_dir, model_dir) for model_dir in model_dirs]
    training_loss_paths = [os.path.join(model_path, f"train_loss_{mRNA_max_len}.json") for model_path in model_paths]
    test_acc_paths = [os.path.join(model_path, f"test_accuracy_{mRNA_max_len}.json") for model_path in model_paths]
    
    if model_names is None:
        # if model_names are not given, model_names takes the folder name of the model
        model_names = model_dirs
    
    plt.figure(figsize=(9,6))
    for json_file, method_name in zip(training_loss_paths, model_names):
        with open(json_file, "r", encoding='utf-8') as f:
            training_loss = json.load(f)
        
        epochs = range(len(training_loss))
        plt.plot(epochs, training_loss, label=f"{method_name}")
    
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    
    # save or show the plot
    if training_loss_save_path:
        plt.savefig(training_loss_save_path, dpi=500, bbox_inches="tight")
        print(f"Training loss plot is saved to {training_loss_save_path}")
    else:
        plt.show()
    
    plt.figure(figsize=(9,6))
    for json_file, method_name in zip(test_acc_paths, model_names):
        with open(json_file, "r", encoding='utf-8') as f:
            test_acc = json.load(f)
        max_accuracy = max(test_acc)
        epochs = range(len(test_acc))
        plt.plot(epochs, test_acc, label=f"{method_name} (Highest = {max_accuracy:.2f})")
    
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    # save or show the plot
    if test_acc_save_path:
        plt.savefig(test_acc_save_path, dpi=500, bbox_inches="tight")
        print(f"Test accuracy plot is saved to {test_acc_save_path}")
    else:
        plt.show()
    
            
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Plot training loss and test accuracy of models.")
    argparser.add_argument(
        "--mRNA_max_len",
        type=int,
        default=1000,
        help="Maximum length of mRNA sequences (default: 1000)"
    )
    argparser.add_argument(
        "--dataset_name",
        type=str,
        default="train dataset",
        help="The name of the folder that indicate which training dataset is used"
    )
    argparser.add_argument(
        "--model_dirs",
        nargs='+',
        help="Name of the folder to each model of which performance is plotted"
    )
    argparser.add_argument(
        "--model_names_in_plot",
        nargs='+',
        required=False,
        help="Name of the model to be used in the plot legend"
    )
    argparser.add_argument(
        "--train_loss_save_path",
        type=str,
        required=False,
        help="Path to save training loss plot"
    )
    argparser.add_argument(
        "--test_acc_save_path",
        type=str,
        required=False,
        help="Path to save test accuracy plot"
    )
    args = argparser.parse_args()
    
    mRNA_max_len = args.mRNA_max_len
    dataset_name = args.dataset_name
    model_dirs = args.model_dirs
    model_names = args.model_names_in_plot
    train_loss_save_path = args.train_loss_save_path
    test_acc_save_path = args.test_acc_save_path
    
    plot_performance(
        mRNA_max_len=mRNA_max_len,
        dataset_name=dataset_name,
        model_dirs=model_dirs,
        model_names=model_names,
        training_loss_save_path=train_loss_save_path,
        test_acc_save_path=test_acc_save_path,
        )