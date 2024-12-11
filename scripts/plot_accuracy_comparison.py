import os
import matplotlib.pyplot as plt
import numpy as np
import json

# # test accuracy on mirTarBase dataset
# linear_test_accuracy = os.path.join(os.path.expanduser("~/projects/mirLM"), "test_accuracy_500_2MLP_revmasked.json")
# finetune_Hyena_accuracy = os.path.join(os.path.expanduser("~/projects/mirLM"), "test_accuracy_500_finetune_Hyena.json")
# cnn_accuracy = os.path.join(os.path.expanduser("~/projects/mirLM"), "test_accuracy_CNN_mirLM_data.json")

# with open(linear_test_accuracy, "r") as fp:
#     linear_test_accuracy = json.load(fp)
    
# with open(finetune_Hyena_accuracy, "r") as fp:
#     finetune_Hyena_accuracy = json.load(fp)

# with open(cnn_accuracy, "r") as fp:
#     cnn_accuracy = json.load(fp)

# # print("length of linear accuracy = ", len(linear_test_accuracy))
# # print("length of finetuned hyena accuracy = ", len(finetune_Hyena_accuracy))
# # print("length of cnn accuracy = ", len(cnn_accuracy))

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
# ax.plot(np.arange(len(linear_test_accuracy)), linear_test_accuracy, 'o-', color='orange', label='Trained 3-layer MLP')
# ax.plot(np.arange(len(finetune_Hyena_accuracy[:50])), finetune_Hyena_accuracy[:50], 'o-', color='green', label='Finetuned Hyena + 3-layer MLP')
# ax.plot(np.arange(len(cnn_accuracy)), cnn_accuracy, 'o-', color='blue', label='Trained CNN + 3-layer MLP')
# ax.set_xlabel("epoch")
# ax.set_ylabel("Accuracy (%)")
# ax.set_title("Test accuracy vs epochs")
# ax.legend()

# plot_path = os.path.join(os.path.expanduser("~/projects/mirLM"), "Peformance_500_epochs_50_accuracy_comparison.png")

# plt.savefig(plot_path)

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

# test accuracy on 1000/2000/3000/4000-nt mirTarBase dataset
Perf_dir = os.path.expanduser("~/projects/mirLM/Performance/mirLM")
mRNA_length = 5000
epochs = 10
# linear_test_accuracy = os.path.join(Perf_dir, "MLP", f"test_accuracy_{mRNA_length}_2MLP_revmasked.json")
linear_test_accuracy = [62.38, 62.65, 62.48, 62.39, 63.96, 63.69, 63.72, 64.30, 64.32, 64.25]
# linear_test_accuracy = [63.64] + [np.nan]*9
# linear_test_accuracy = [62.86] + [np.nan]*9
# linear_test_accuracy = [62.40] + [np.nan]*9
linear_test_accuracy = [63.73] + [np.nan]*9
finetune_Hyena_accuracy = os.path.join(Perf_dir, "HyenaDNA", f"test_accuracy_{mRNA_length}.json")
cnn_accuracy = os.path.join(Perf_dir, "CNN", f"test_accuracy_{mRNA_length}.json")

# with open(linear_test_accuracy, "r") as fp:
    # linear_test_accuracy = json.load(fp)
    
with open(finetune_Hyena_accuracy, "r") as fp:
    finetune_Hyena_accuracy = json.load(fp)

with open(cnn_accuracy, "r") as fp:
    cnn_accuracy = json.load(fp)

print("length of linear accuracy = ", len(linear_test_accuracy))
print("length of finetuned hyena accuracy = ", len(finetune_Hyena_accuracy))
print("length of cnn accuracy = ", len(cnn_accuracy))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
ax.plot(np.arange(len(linear_test_accuracy)), linear_test_accuracy, 'o-', color='orange', label='Trained 3-layer MLP')
ax.plot(np.arange(len(finetune_Hyena_accuracy)), finetune_Hyena_accuracy, 'o-', color='green', label='Finetuned Hyena + 3-layer MLP')
ax.plot(np.arange(len(cnn_accuracy)), cnn_accuracy, 'o-', color='blue', label='Trained CNN + 3-layer MLP')
ax.set_xlabel("epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Test accuracy vs epochs")
ax.legend()

plot_path = os.path.join(os.path.expanduser("~/projects/mirLM"), f"Peformance_{mRNA_length}_epochs_{epochs}_accuracy_comparison.png")

plt.savefig(plot_path)
