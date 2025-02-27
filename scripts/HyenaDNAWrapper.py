import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
# Local imports
from mirLM import mirLM

class HyenaDNAWrapper(mirLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def run_training(
        self,
        model,
        train_loader,
        optimizer,
        epoch,
        loss_fn,
        log_interval=10,
    ):
        """
        Training loop.
        """
        model.train()
        epoch_loss = 0.0
        loss_list = []
        optimizer.zero_grad()
        for batch_idx, (seq, seq_mask, target) in enumerate(train_loader):
            seq, seq_mask, target = (
                seq.to(self.device), 
                seq_mask.to(self.device),
                target.to(self.device),
            )
            output = self.forward(seq=seq, 
                                  seq_mask=seq_mask,
                                 )
            output = self.forward(seq=seq, seq_mask=seq_mask)
            loss = loss_fn(output.squeeze().sigmoid(), target.squeeze())
            if self.accumulation_step is not None:
                loss = loss / self.accumulation_step
                loss.backward()
                loss_list.append(loss.item())
                if (batch_idx + 1) % self.accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if self.ddp:
                        print(
                                f"[Rank {dist.get_rank()}] "
                                f"Train Epoch: {epoch} "
                                f"[{(batch_idx + 1) * len(seq)}/{len(train_loader.sampler)} "
                                f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                                f"Avg Loss: {sum(loss_list) / len(loss_list):.6f}\n",
                                flush=True
                            )
                    else:
                        print(
                                f"Train Epoch: {epoch} "
                                f"[{(batch_idx + 1) * len(seq)}/{len(train_loader.dataset)} "
                                f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                                f"Avg Loss: {sum(loss_list) / len(loss_list):.6f}\n",
                                flush=True
                            )
                    loss_list = []
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (batch_idx + 1) % log_interval == 0:
                    if self.ddp:
                        print(
                                f"[Rank {dist.get_rank()}] "
                                f"Train Epoch: {epoch} "
                                f"[{(batch_idx + 1) * len(seq)}/{len(train_loader.dataset) // int(os.environ['WORLD_SIZE'])} "
                                f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                                f"Loss: {loss.item():.6f}\n"
                            )
                    else:
                        print(
                                f"Train Epoch: {epoch} "
                                f"[{(batch_idx + 1) * len(seq)}/{len(train_loader.dataset)} "
                                f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                                f"Loss: {loss.item():.6f}\n"
                            )                        
            epoch_loss += loss.item() * self.accumulation_step
        # After the loop, if gradients remain (for non-divisible number of batches)
        if (batch_idx + 1) % self.accumulation_step != 0:
            optimizer.step()
            optimizer.zero_grad()
        average_loss = epoch_loss / len(train_loader)
        return average_loss 
    
    def run_testing(
        self, 
        model,
        test_loader, 
        loss_fn,
    ):
        """Test loop."""
        model.eval()
        losses = []
        if self.ddp:
            local_correct = 0
        else:
            correct = 0
        with torch.no_grad():
            for seq, seq_mask, target in test_loader:
                seq, target, seq_mask = (
                    seq.to(self.device),
                    target.to(self.device),
                    seq_mask.to(self.device),
                )
                output = self.forward(seq=seq, seq_mask=seq_mask)
                loss = loss_fn(output.squeeze().sigmoid(), target.squeeze())
                losses.append(loss.item())
                probabilities = torch.sigmoid(output.squeeze())
                predictions = (probabilities > 0.5).long()
                if self.ddp:
                    local_correct += predictions.eq(target.squeeze()).sum().item()
                else:
                    correct += predictions.eq(target.squeeze()).sum().item()
        
        if self.ddp:
            # convert to gpu tensor
            correct_tensor = torch.tensor(local_correct, dtype=torch.long, device=self.device)
            # Sum across all ranks
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            # Get all correct counts
            global_correct = correct_tensor.item()
            # compute global accuracy
            global_accuracy = 100.0 * global_correct / len(test_loader.dataset)
            if dist.get_rank() == 0:
                print(
                    f"Test set: Accuracy: {global_correct}/{len(test_loader.dataset)} "
                    f"({global_accuracy:.2f}%)\n"
                )
            return global_accuracy
        else:
            avg_loss = sum(losses) / len(losses)
            accuracy = 100.0 * correct / len(test_loader.dataset)
            print(
                "\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(
                    correct, len(test_loader.dataset), accuracy
                )
            )
            return accuracy, avg_loss

    @staticmethod
    def assess_acc(predictions, targets, thresh=0.5):
        y = np.asarray(targets)
        y_hat = np.asarray(predictions)
        y_hat = np.uint8(y_hat > thresh)
        correct = np.sum(y == y_hat)
        acc = correct / len(y)
        return acc
    
    def run_evaluation(
        self, 
        model,
        test_loader, 
        ):
        """Test loop with ddp."""
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for seq, seq_mask, target in test_loader:
                seq, target, seq_mask = (
                    seq.to(self.device),
                    target.to(self.device),
                    seq_mask.to(self.device),
                )
                output = model.forward(seq=seq, seq_mask=seq_mask, train=False)
                probabilities = torch.sigmoid(output.squeeze()).cpu().numpy().tolist()
                # record predictions
                targets = target.cpu().view(-1).numpy().tolist()
                predictions.extend(probabilities)
                true_labels.extend(targets)
            
        acc = self.assess_acc(predictions=predictions, targets=true_labels)
        return acc, predictions, true_labels
    
    def forward(self, seq, seq_mask):
        """
        Forward pass for HyenaDNAWrapper.
        """
        seq = seq.long()
        seq_mask = seq_mask.long()
        return self.hyena(
            input_ids=seq,
            input_mask=seq_mask,
            use_only_miRNA=True,
            add_linker=True,
            max_mRNA_length=self.mRNA_max_len,
            max_miRNA_length=self.miRNA_max_len
        )  
        
       
