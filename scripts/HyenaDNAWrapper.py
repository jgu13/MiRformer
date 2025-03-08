import os
import math
import json
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from transformers.tokenization_utils import PreTrainedTokenizer
# Local imports
from mirLM import mirLM

class HyenaDNAWrapper(mirLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def generate_perturbed_seeds(
                            self,
                            original_seq, 
                            original_mask,
                            seed_start, 
                            seed_end,
                            tokenizer):
        special_chars = ["[PAD]","[UNK]","[MASK]"]
        perturbed_seqs = []
        perturbed_masks = []
        original_seq = [tokenizer._convert_id_to_token(d.item()) for d in original_seq]
        seed_start = int(seed_start.item()) if torch.is_tensor(seed_start) else int(seed_start)
        seed_end = int(seed_end.item()) if torch.is_tensor(seed_end) else int(seed_end)
        seed_match = original_seq[seed_start:seed_end]
        for i in range(len(seed_match)):
            if seed_match[i] in special_chars:
                continue
            for base in ["A", "T", "C", "G"]:
                if base != seed_match[i]:
                    mutated = seed_match[:i] + [base] + seed_match[i+1:]
                    perturbed_seq = original_seq[:seed_start] + mutated + original_seq[seed_end:]
                    # convert back to ids
                    perturbed_seq = [tokenizer._convert_token_to_id(c) for c in perturbed_seq]
                    perturbed_seqs.append(torch.tensor(perturbed_seq, dtype=torch.long))
                    perturbed_masks.append(original_mask)
        return perturbed_seqs, perturbed_masks # 3 * seed_len
    
    def run_training(
        self,
        model,
        train_loader,
        optimizer,
        epoch,
        loss_fn,
        tokenizer,
        log_interval=10,
        margin=0.1,
        alpha=0.1,
    ):
        """
        Training loop.
        """
        wandb.login(key="600e5cca820a9fbb7580d052801b3acfd5c92da2")
        wandb.init(project="mirLM_study", config={
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "alpha": self.alpha,
            "margin": self.margin,
            # add any other hyperparameters you want to track
        })
        config = wandb.config
        
        model.train()
        epoch_loss = 0.0
        loss_list = []
        optimizer.zero_grad()
        for batch_idx, (seq, seq_mask, seed_start, seed_end, target) in enumerate(train_loader):
            seq, seq_mask, seed_start, seed_end, target = (
                seq.to(self.device), 
                seq_mask.to(self.device),
                seed_start.to(self.device),
                seed_end.to(self.device),
                target.to(self.device),
            )
            pos_mask = (target == 1)
            pos_indices = torch.where(pos_mask)[0]
            if pos_indices.shape[0] > 0:
                s_w, s_i, repeats = model.forward(seq=seq, 
                                                seq_mask=seq_mask,
                                                seed_start=seed_start,
                                                seed_end=seed_end,
                                                tokenizer=tokenizer,
                                                pos_indices=pos_indices,
                                                perturb=True,
                                                )
                s_w, s_i = (s_w.squeeze(-1), s_i.squeeze(-1))
                bce_loss = loss_fn(s_w, target.view(-1))
                s_w_pos = s_w[pos_indices] # repeat only the positive samples
                s_w_pos_repeated = torch.repeat_interleave(s_w_pos, repeats) # (batchsize * seed_len * 3,)
                # ranking loss = -ylog(\sigma(s_w-s_i-margin)) where y = 1, enforcing s_w - s_i > margin
                difference = s_w_pos_repeated - s_i - margin
                ranking_loss = loss_fn(difference, torch.ones_like(difference))
                total_loss = bce_loss + alpha * ranking_loss
                # Log individual losses to wandb
                wandb.log({
                    "training_bce_loss": bce_loss.item(),
                    "training_ranking_loss": ranking_loss.item(),
                    "training_total_loss": total_loss.item(),
                })
            else: # no positive samples in this batch
                s_w = model.forward(seq=seq,
                                    seq_mask=seq_mask,
                                    perturb=False)
                total_loss = loss_fn(s_w.squeeze(), target.view(-1)) # loss equals to bce loss when no perturbation

            if self.accumulation_step is not None:
                total_loss = total_loss / self.accumulation_step
                total_loss.backward()
                loss_list.append(total_loss.item())
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
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (batch_idx + 1) % log_interval == 0:
                    if self.ddp:
                        print(
                                f"[Rank {dist.get_rank()}] "
                                f"Train Epoch: {epoch} "
                                f"[{(batch_idx + 1) * len(seq)}/{len(train_loader.dataset) // int(os.environ['WORLD_SIZE'])} "
                                f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                                f"Loss: {total_loss.item():.6f}\n"
                            )
                    else:
                        print(
                                f"Train Epoch: {epoch} "
                                f"[{(batch_idx + 1) * len(seq)}/{len(train_loader.dataset)} "
                                f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                                f"Loss: {total_loss.item():.6f}\n"
                            )                        
            epoch_loss += total_loss.item() * self.accumulation_step
        # After the loop, if gradients remain (for non-divisible number of batches)
        if (batch_idx + 1) % self.accumulation_step != 0:
            optimizer.step()
            optimizer.zero_grad()
        average_loss = epoch_loss / len(train_loader)
        wandb.log({
            "epoch_loss": average_loss,
        })
        return average_loss 
    
    def run_testing(
        self, 
        model,
        test_loader,
        tokenizer,
        loss_fn,
        alpha,
        margin,
    ):
        """Test loop."""
        model.eval()
        losses = []
        if self.ddp:
            local_correct = 0
        else:
            correct = 0
        unperturbed_predictions = []
        perturbed_predictions = []
        with torch.no_grad():
            for seq, seq_mask, seed_start, seed_end, target in test_loader:
                seq, seq_mask, seed_start, seed_end, target = (
                    seq.to(self.device),
                    seq_mask.to(self.device),
                    seed_start.to(self.device),
                    seed_end.to(self.device),
                    target.to(self.device),
                )
                pos_mask = (target == 1)
                pos_indices = torch.where(pos_mask)[0]
                if pos_indices.shape[0] > 0:
                    s_w, s_i, repeats = model.forward(seq=seq, 
                                                seq_mask=seq_mask,
                                                seed_start=seed_start,
                                                seed_end=seed_end,
                                                tokenizer=tokenizer,
                                                pos_indices=pos_indices,
                                                perturb=True,
                                                )
                    s_w, s_i = (s_w.squeeze(-1), s_i.squeeze(-1))
                    bce_loss = loss_fn(s_w, target.view(-1))
                    s_w_pos = s_w[pos_indices] # repeat only the positive samples
                    s_w_pos_repeated = torch.repeat_interleave(s_w_pos, repeats) # (batchsize * seed_len * 3,)
                    difference = s_w_pos_repeated - s_i - margin
                    ranking_loss = loss_fn(difference, torch.ones_like(difference))
                    total_loss = bce_loss + alpha * ranking_loss
                    losses.append(total_loss.item())
                    # Log individual losses to wandb
                    wandb.log({
                        "evaluation_bce_loss": bce_loss.item(),
                        "evaluation_ranking_loss": ranking_loss.item(),
                        "evaluation_total_loss": total_loss.item(),
                    })
                else: 
                    s_w = model.forward(seq=seq,
                                        seq_mask=seq_mask,
                                        perturb=False,
                                        )
                    s_w = s_w.squeeze(-1)
                    total_loss = loss_fn(s_w, target.view(-1)) # total loss equals bce loss when no perturbation
                    losses.append(total_loss.item())
                prediction = (s_w > 0.5).long() # (batchsize,)
                if self.ddp:
                    local_correct += prediction.eq(target.view(-1)).sum().item()
                else:
                    correct += prediction.eq(target.view(-1)).sum().item()
                unperturbed_predictions.extend(s_w_pos_repeated.cpu())
                perturbed_predictions.extend(s_i.cpu())
            # change in prediction
            mean_unperturbed_score = np.mean(np.asarray(unperturbed_predictions))
            mean_perturbed_score = np.mean(np.asarray(perturbed_predictions))
            diff_score = (mean_unperturbed_score - mean_perturbed_score).item()
            # wandb.log({
            #     "Mean_unpertubed_score": mean_unperturbed_score,
            #     "Mean_perturbed_score": mean_perturbed_score,
            # })
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
            accuracy = 100.0 * correct / len(test_loader.dataset)
            avg_loss = sum(losses) / len(losses)
            print(
                "\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(
                    correct, len(test_loader.dataset), accuracy
                )
            )
            print(f"Average unperturbed prediction: {mean_unperturbed_score:.4f}")
            print(f"Average perturbed prediction: {mean_perturbed_score:.4f}")
            print(f"Difference (unperturbed - perturbed): {diff_score:.4f}")
            # Log individual losses to wandb
            wandb.log({
                "Accuracy": accuracy,
            })
            return accuracy, diff_score, avg_loss

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
            for seq, seq_mask, seed_start, seed_end, target in test_loader:
                seq, target, seq_mask = (
                    seq.to(self.device),
                    target.to(self.device),
                    seq_mask.to(self.device),
                )
                output = model.forward(seq=seq, seq_mask=seq_mask, perturb=False)
                probabilities = torch.sigmoid(output.squeeze()).cpu().numpy().tolist()
                # record predictions
                targets = target.cpu().view(-1).numpy().tolist()
                predictions.extend(probabilities)
                true_labels.extend(targets)
            
        acc = self.assess_acc(predictions=predictions, targets=true_labels)
        return acc, predictions, true_labels
    
    def forward(self, 
                seq:torch.tensor, 
                seq_mask:torch.tensor,
                seed_start:torch.tensor=-1,
                seed_end:torch.tensor=-1,
                pos_indices:torch.tensor=None,
                tokenizer:PreTrainedTokenizer=None,
                perturb=True):
        """
        Forward pass for HyenaDNAWrapper.
        """
        seq = seq.long()
        seq_mask = seq_mask.long()
        s_w = self.hyena(
            input_ids=seq,
            input_mask=seq_mask,
            use_only_miRNA=True,
            add_linker=True,
            max_mRNA_length=self.mRNA_max_len,
            max_miRNA_length=self.miRNA_max_len
        )
        if perturb:
            assert (pos_indices is not None) and (pos_indices.shape[0]>0), "Only Positive samples will be perturbed during training. Positive samples are not provided."
            assert tokenizer is not None, "Tokenizer to tokenize perturbed sequences. Tokenizer can not be None."
            
            perturbed_seqs = []
            perturbed_masks = []
            repeats = [] # the number of time to repeat s_w
            for idx in pos_indices:
                original_seq = seq[idx]
                original_mask = seq_mask[idx]
                start = seed_start[idx]
                end = seed_end[idx]
                assert (start.item() > 0) and (end.item() > 0), "During training, seed start and seed end must be provided to perturb the seed region."
                # Generate perturbed sequences for the seed region
                perturbed, perturbed_mask = self.generate_perturbed_seeds(
                        original_seq=original_seq, 
                        original_mask=original_mask,
                        seed_start=start, 
                        seed_end=end,
                        tokenizer=tokenizer)
                perturbed_seqs.extend(perturbed)
                perturbed_masks.extend(perturbed_mask)
                repeats.append(len(perturbed))
                # print("repeats = ", repeats)
            repeats = torch.tensor(repeats, dtype=torch.long, device=self.device)
            # perturbed_seqs has shape (batchsize * seed_len * 3)
            # Compute perturbed scores (s_i)
            s_i = self.hyena(
                input_ids = torch.stack(perturbed_seqs).to(self.device),
                input_mask = torch.stack(perturbed_masks).to(self.device),
                use_only_miRNA=True,
                add_linker=True,
                max_mRNA_length=self.mRNA_max_len,
                max_miRNA_length=self.miRNA_max_len
                ) # (batchsize * seed_len * 3, 1) 
            return s_w, s_i, repeats 
        else:
            return s_w
