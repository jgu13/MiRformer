import os
import matplotlib.pyplot as plt
import numpy as np
import json

# test accuracy on mirTarBase dataset
linear_test_accuracy = os.path.join(os.path.expanduser("~/projects/mirLM/Performance/mirLM/MLP"), "test_accuracy_1000.json")
finetune_Hyena_accuracy = os.path.join(os.path.expanduser("~/projects/mirLM/Performance/mirLM/HyenaDNA"), "test_accuracy_1000_continued_training.json")
cnn_accuracy = os.path.join(os.path.expanduser("~/projects/mirLM/Performance/mirLM/CNN"), "test_accuracy_1000.json")

with open(linear_test_accuracy, "r") as fp:
    linear_test_accuracy = json.load(fp)

with open(finetune_Hyena_accuracy, "r") as fp:
    finetune_Hyena_accuracy = json.load(fp)

with open(cnn_accuracy, "r") as fp:
    cnn_accuracy = json.load(fp)

print("length of linear accuracy = ", len(linear_test_accuracy))
print("length of finetuned hyena accuracy = ", len(finetune_Hyena_accuracy))
print("length of cnn accuracy = ", len(cnn_accuracy))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
ax.plot(np.arange(len(linear_test_accuracy)), linear_test_accuracy, 'o-', color='orange', label='HyenaDNA-MLP-CrossAttn')
ax.plot(np.arange(len(finetune_Hyena_accuracy[:50])), finetune_Hyena_accuracy[:50], 'o-', color='green', label='Finetuned HyenaDNA-MLP')
ax.plot(np.arange(len(cnn_accuracy)), cnn_accuracy, 'o-', color='blue', label='baseline CNN-MLP')
ax.set_xlabel("epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Test accuracy vs epochs on mRNA = 1000nts")
ax.legend()

plot_path = os.path.join(os.path.expanduser("~/projects/mirLM"), "Peformance_1000_epochs_50_accuracy_comparison.png")

plt.savefig(plot_path)

# test accuracy on miRAW dataset
# linear_test_accuracy = os.path.join(os.path.expanduser("~/projects/mirLM"), "test_accuracy_miRAW_2MLP_revmasked.json")
# finetune_Hyena_accuracy = os.path.join(os.path.expanduser("~/projects/mirLM"), "test_accuracy_miRAW_finetune_Hyena.json")
# cnn_accuracy = os.path.join(os.path.expanduser("~/projects/mirLM"), "test_accuracy_CNN_miRAW_batchsize_256.json")

# with open(linear_test_accuracy, "r") as fp:
#     linear_test_accuracy = json.load(fp) # epoch 50

# with open(finetune_Hyena_accuracy, "r") as fp:
#     finetune_Hyena_accuracy = json.load(fp) # epoch 100

# with open(cnn_accuracy, "r") as fp:
#     cnn_accuracy = json.load(fp) # epoch 100

# lower_bound = len(linear_test_accuracy)

# # print("length of linear accuracy = ", len(linear_test_accuracy))
# # print("length of finetuned hyena accuracy = ", len(finetune_Hyena_accuracy))
# # print("length of cnn accuracy = ", len(cnn_accuracy))

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
# ax.plot(np.arange(len(linear_test_accuracy)), linear_test_accuracy, 'o-', color='orange', label='Trained 3-layer MLP')
# ax.plot(np.arange(len(finetune_Hyena_accuracy[:lower_bound])), finetune_Hyena_accuracy[:lower_bound], 'o-', color='green', label='Finetuned Hyena + 3-layer MLP')
# ax.plot(np.arange(len(cnn_accuracy[:lower_bound])), cnn_accuracy[:lower_bound], 'o-', color='blue', label='Trained CNN + 3-layer MLP')
# ax.set_xlabel("epoch")
# ax.set_ylabel("Accuracy (%)")
# ax.set_title("Test accuracy vs epochs")
# ax.legend()

# plot_path = os.path.join(os.path.expanduser("~/projects/mirLM"), "Peformance_miRAW_epochs_50_accuracy_comparison.png")

# plt.savefig(plot_path)

# # test accuracy on 1000/2000/3000/4000-nt mirTarBase dataset
# Perf_dir = os.path.expanduser("~/projects/mirLM/Performance/mirLM")
# mRNA_length = 4000
# epochs = 9
# # linear_test_accuracy = os.path.join(Perf_dir, "MLP", f"test_accuracy_{mRNA_length}_2MLP_revmasked.json")
# linear_test_accuracy_lr_3e4 = [
#     57.06, 58.06, 60.33, 60.07, 60.93, 59.73, 60.56, 62,    62.17
# ]
# linear_test_accuracy_lr_6e5 = [
#     58.25, 59.54, 60.85, 61.71, 61.68, 58.25, 59.54, 60.85, 61.71
# ]

# # finetune_Hyena_accuracy = os.path.join(
# #     Perf_dir, "HyenaDNA", f"test_accuracy_{mRNA_length}.json"
# # )
# # cnn_accuracy = os.path.join(Perf_dir, "CNN", f"test_accuracy_{mRNA_length}.json")
# cnn_accuracy_lr_3e4 = [
#     65.47, 65.04, 63.77, 63.95, 63.75, 63.49, 63.70, 63.59, 64.18
# ]
# cnn_accuracy_lr_6e5 = [
#     65.68, 65.79, 65.55, 65.63, 65.68, 65.66, 65.77, 65.71, 65.56
# ]

# # with open(linear_test_accuracy, "r") as fp:
# # linear_test_accuracy = json.load(fp)

# # with open(finetune_Hyena_accuracy, "r") as fp:
# #     finetune_Hyena_accuracy = json.load(fp)

# # with open(cnn_accuracy, "r") as fp:
# #     cnn_accuracy = json.load(fp)

# # print("length of linear accuracy = ", len(linear_test_accuracy))
# # print("length of finetuned hyena accuracy = ", len(finetune_Hyena_accuracy))
# # print("length of cnn accuracy = ", len(cnn_accuracy))

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
# ax.plot(
#     np.arange(len(linear_test_accuracy_lr_3e4)),
#     linear_test_accuracy_lr_3e4,
#     "o-",
#     color="orange",
#     label="Trained on lr=3e-4",
# )
# ax.plot(
#     np.arange(len(linear_test_accuracy_lr_6e5)),
#     linear_test_accuracy_lr_6e5,
#     "o-",
#     color="green",
#     label="Trained on lr=6e-5",
# )
# ax.set_xlabel("epoch")
# ax.set_ylabel("Accuracy (%)")
# ax.set_title("MLP Test Accuracy")
# ax.legend()

# plot_path = os.path.join(
#     os.path.expanduser("~/projects/mirLM"),
#     f"Peformance_{mRNA_length}_epochs_{epochs}_accuracy_comparison_MLP_diff_learning_rate.png",
# )

# plt.savefig(plot_path)
# # ax.plot(
# #     np.arange(len(finetune_Hyena_accuracy)),
# #     finetune_Hyena_accuracy,
# #     "o-",
# #     color="green",
# #     label="Finetuned Hyena + 3-layer MLP",
# # )

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
# ax.plot(
#     np.arange(len(cnn_accuracy_lr_3e4)),
#     cnn_accuracy_lr_3e4,
#     "o-",
#     color="blue",
#     label="Trained on lr=3e-4",
# )
# ax.plot(
#     np.arange(len(cnn_accuracy_lr_6e5)),
#     cnn_accuracy_lr_6e5,
#     "o-",
#     color="green",
#     label="Trained on lr=6e-5",
# )
# ax.set_xlabel("epoch")
# ax.set_ylabel("Accuracy (%)")
# ax.set_title("CNN Test Accuracy")
# ax.legend()

# plot_path = os.path.join(
#     os.path.expanduser("~/projects/mirLM"),
#     f"Peformance_{mRNA_length}_epochs_{epochs}_accuracy_comparisonMLP_CNN_diff_learning_rate.png",
# )

# plt.savefig(plot_path)
