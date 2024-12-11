import torch.nn.functional as F
import torch

x = torch.tensor([[-9.0, 9.0]])
target = torch.tensor([[0.0, 1.0]])

print("x shape = ", x.size())
print("target shape = ", target.size())
ce = F.cross_entropy(x, target=torch.LongTensor([1]))
bce = F.binary_cross_entropy(torch.sigmoid(x), target)
bce_w_logits = F.binary_cross_entropy_with_logits(x, target)

print("Cross entropy loss = ", ce)
print("Binary cross entropy loss = ", bce)
print("Binary cross entropy loss w logits = ", bce_w_logits)
