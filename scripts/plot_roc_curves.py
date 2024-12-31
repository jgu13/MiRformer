import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse

def plot_auc(json_files, method_names=None, output_file=None):
    """
    Load prediction/true-label pairs from a list of JSON files,
    compute ROC curves, and plot them on a single figure with AUROC in the legend.

    :param json_files: List of paths to JSON files. Each file must have:
                       {
                           "true_labels": [0 or 1,  ...],
                           "predictions": [prob_for_positive_class, ...]
                       }
    :param method_names: List of method names to use in the legend. If None,
                         it will default to enumerated method labels.
    :param output_file: path to save plot.
    """
    if method_names is None:
        # If no method names provided, label them generically
        method_names = [f"Method {i+1}" for i in range(len(json_files))]

    plt.figure(figsize=(6, 6))

    # Iterate over JSON files and corresponding method names
    for json_file, method_name in zip(json_files, method_names):
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract ground truth labels and predicted probabilities
        y_true = np.array(data["true_labels"])
        y_score = np.array(data["predictions"])
               
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for this method
        plt.plot(fpr, tpr, label=f"{method_name} (AUC = {roc_auc:.3f})")

    # Plot the diagonal "random guess" line
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Guess")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparison of Classification Methods (ROC Curve)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    """
    Main function to parse command-line arguments and call plot_auc.
    Usage:
        python plot_auc.py --json_files method1_predictions.json method2_predictions.json ...
    """
    parser = argparse.ArgumentParser(
        description="Plot ROC curves (with AUROC) for multiple classifiers from JSON files."
    )
    parser.add_argument(
        "--json_files",
        nargs="+",
        required=True,
        help="Paths to JSON files, each containing 'labels' and 'predictions'."
    )
    parser.add_argument(
        "--method_names",
        nargs="+",
        required=False,
        help="Names of the methods to display in the legend. Defaults to Method 1, Method 2, ..."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        help="Filepath to save the plot (e.g., 'roc_plot.png'). If not provided, the plot will be displayed."
    )
    args = parser.parse_args()

    # Call the plotting function
    plot_auc(args.json_files, args.method_names, args.output_file)


if __name__ == "__main__":
    main()

