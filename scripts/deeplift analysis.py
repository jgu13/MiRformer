import shap
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deeplift.visualization import viz_sequence
from deeplift.dinuc_shuffle import dinuc_shuffle

def load_model(mRNA_max_len,
               miRNA_max_len,
               device,
               model_name,
               dataset_name):
    
    if model_name == 'CNN':
    
    elif model_name == 'TwoTowerMLP':
        
    elif model_name == 'HyenaDNA':
        
    elif model_name == 'TwoTowerMLP_Attn':
    
    elif model_name == 'Attn':
    

def shuffle_several_times(s):
    s = np.squeeze(s)
    return dinuc_shuffle(s, num_shufs=100)

def deeplift_analysis(model, encoded_sequences_test, labels_test, device):
    """
    Implements DeepLIFT via shap.DeepExplainer to visualize the test sequences that the model
    used for the classification. Provides visualizations for the most confident correct and
    incorrect predictions for each class.
    """

    np.random.seed(1)
    torch.manual_seed(1)

    model.eval()

    # Move data to device
    X_test = encoded_sequences_test.to(device)
    y_test = labels_test.to(device)

    # Get model predictions and probabilities
    with torch.no_grad():
        outputs = model(X_test)
        probabilities = nn.functional.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)

    # Move tensors to CPU for processing
    X_test_cpu = X_test.cpu().numpy()
    y_test_cpu = y_test.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    confidences_cpu = confidences.cpu().numpy()

    # Number of classes
    num_classes = len(np.unique(y_test_cpu))

    # Initialize a dictionary to store selected indices
    selected_indices = []

    # For each class, find the most confident correct and incorrect predictions
    for true_cls in range(num_classes):
        for pred_cls in range(num_classes):
            indices = np.where((y_test_cpu == true_cls) & (predictions_cpu == pred_cls))[0]
            if len(indices) > 0:
                # Select the index with the highest confidence
                cls_confidences = confidences_cpu[indices]
                most_confident_idx = indices[np.argmax(cls_confidences)]
                selected_indices.append((true_cls, pred_cls, most_confident_idx))

    # Prepare the background function using dinuc_shuffle
    def background_function(s):
        return np.array([shuffle_several_times(seq) for seq in s])

    # Initialize plots
    fig, axes = plt.subplots(num_classes, num_classes, figsize=(15, 15))

    # For each selected sequence, compute SHAP values and visualize
    for true_cls in range(num_classes):
        for pred_cls in range(num_classes):
            # Find the corresponding index
            idx_tuple = next((t for t in selected_indices if t[0] == true_cls and t[1] == pred_cls), None)
            ax = axes[true_cls, pred_cls]
            if idx_tuple is not None:
                idx = idx_tuple[2]
                sequence = X_test_cpu[idx]  # Shape: (4, sequence_length)
                sequence_label = y_test_cpu[idx]
                predicted_label = predictions_cpu[idx]
                confidence = confidences_cpu[idx]

                # Prepare background data by shuffling the sequence multiple times
                background = background_function(sequence[np.newaxis, ...])

                # Define model forward function
                def model_forward(x):
                    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        outputs = model(x_tensor)
                        outputs = nn.functional.softmax(outputs, dim=1)
                    return outputs.cpu().numpy()

                # Initialize DeepExplainer
                dinuc_shuff_explainer = shap.DeepExplainer(model_forward, background)

                # Compute SHAP values
                raw_shap_values = dinuc_shuff_explainer.shap_values(sequence[np.newaxis, ...], check_additivity=False)
                shap_values = raw_shap_values[predicted_label][0]  # Get shap values for the predicted class

                # Project the importance onto the base that's actually present
                # shap_values is of shape (4, sequence_length)
                # sequence is of shape (4, sequence_length)
                projected_shap_values = np.sum(shap_values, axis=0)[:, None] * sequence.T

                # Plotting
                viz_sequence.plot_weights(projected_shap_values, subticks_frequency=10, ax=ax)
                ax.set_title(f'True: {true_cls}, Pred: {pred_cls}, Conf: {confidence:.2f}')
            else:
                ax.axis('off')
                ax.set_title(f'True: {true_cls}, Pred: {pred_cls}\nNo Sample')

    fig.tight_layout()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Deeplift analysis on test mRNA sequences")
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
        "--test_dataset_path",
        type=str,
        default="path/to/testdataset",
        help="Path to test dataset"
    )
    args = parser.parse_args()
    mRNA_max_len = args.mRNA_max_len
    miRNA_max_len = args.miRNA_max_len
    device = args.device
    model_name = args.model_name
    dataset_name = args.dataset_name
    test_dataset_path = args.test_dataset_path
    
    # load checkpoint of models
    model = load_model(mRNA_max_len=mRNA_max_len,
                        miRNA_max_len=miRNA_max_len,
                        device=device,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        )
    
    
    # perform DeepLIFT analysis
    deeplift_analysis(model, X_test, y_test, device)
    
