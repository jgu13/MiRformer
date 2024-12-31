import os
import matplotlib.pyplot as plt
import numpy as np
import json

avg_loss_path = os.path.join(
    os.path.expanduser("~/projects/mirLM"), "train_loss_500_2MLP_revmasked.json"
)
test_accuracy_path = os.path.join(
    os.path.expanduser("~/projects/mirLM"), "test_accuracy_500_2MLP_revmasked.json"
)

with open(avg_loss_path, "r") as fp:
    avg_loss = json.load(fp)
    avg_loss = np.asarray(avg_loss)

with open(test_accuracy_path, "r") as fp:
    test_accuracy = json.load(fp)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].plot(
    np.arange(len(test_accuracy)),
    test_accuracy,
    "o-",
    color="orange",
    label="Test accuracy",
)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Test accuracy vs epochs")
axes[0].legend()

axes[1].plot(
    np.arange(len(avg_loss)), avg_loss, "o-", color="green", label="Avg training loss"
)
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("Average Loss")
axes[1].set_title("Average Training Loss vs epochs")
axes[1].legend()

plot_path = os.path.join(
    os.path.expanduser("~/projects/mirLM"),
    "Peformance_500_epochs_30_batch_16_2MLP_revmasked.png",
)

plt.savefig(plot_path)
