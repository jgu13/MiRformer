import torch
from collections import OrderedDict

old_checkpoint_path = "/home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/CNN/3000/checkpoint_final.pth"
checkpoint = torch.load(old_checkpoint_path, map_location="cpu")
original_state_dict = checkpoint["model_state_dict"]

# Remove the 'module.' prefix from each key
new_state_dict = OrderedDict()
for k, v in original_state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

# Update the checkpoint dictionary with the corrected state_dict
checkpoint["model_state_dict"] = new_state_dict

# Save the corrected checkpoint
torch.save(checkpoint, 
    "/home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/CNN/3000/checkpoint_final_corrected.pth")