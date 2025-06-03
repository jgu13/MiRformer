import os
import math
import json
import wandb
import numpy as np
import torch
from typing import List
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
# Local imports
from mirLM import mirLM
from Hyena_layer import HyenaDNAPreTrainedModel, HyenaDNAModel, LinearHead

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

class TwoTowerMLP(mirLM):
    def __init__(
        self,
        **kwargs,
    ):
        """
        Args:
            hidden_sizes (list): Hidden sizes for the MLP head.
        """
        super().__init__(
            **kwargs,
        )

        # Initialize the MLP head
        if kwargs.get("basemodel_cfg", ""):
            hidden_size = self.basemodel_cfg.get("hidden_size", "")
            num_hidden_layers = self.basemodel_cfg.get("num_hidden_layers", "")
            if hidden_size and num_hidden_layers:
                hidden_sizes = [hidden_size] * num_hidden_layers
            elif self.basemodel_cfg.get("hidden_sizes", ""):
                hidden_sizes = self.basemodel_cfg["hidden_sizes"]
        elif kwargs.get("hidden_sizes", ""):
            hidden_sizes = kwargs["hidden_sizes"]
        else: # by default, MLP is 2-layer ff with dimension 256
            hidden_sizes = [256] *2
        print("Hidden size = ", hidden_sizes)
        self.mlp_head = LinearHead(
            d_model=self.backbone_cfg["d_model"], 
            d_output=self.n_classes, 
            hidden_sizes=hidden_sizes,
        )

        # Initialize Q and KV hidden_
        self.q_layer = nn.Linear(self.backbone_cfg["d_model"], self.backbone_cfg["d_model"])
        self.kv_layer = nn.Linear(self.backbone_cfg["d_model"], self.backbone_cfg["d_model"])
     
    def compute_cross_attention(
            self,
            Q: torch.tensor, 
            K: torch.tensor, 
            V: torch.tensor, 
            Q_mask: torch.tensor, 
            K_mask: torch.tensor,
            return_attn=False,
        ):
        '''
        Compute cross attention
        '''
        d_model = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)  # [batchsize, miRNA_seq_len, mRNA_seq_len]
        # expand K mask to mask out keys, make each query only attend to valid keys
        K_mask = K_mask.unsqueeze(1).expand(
            -1, Q.shape[1], -1
        )  # [batchsize, mRNA_seq_len, miRNA_seq_len]
        scores = scores.masked_fill(K_mask == 0, -1e9) # [batchsize, miRNA_seq_len, mRNA_seq_len]
        # apply softmax on the key dimension
        attn_weights = F.softmax(scores, dim=-1)  # [batchsize, miRNA_seq_len, mRNA_seq_len]
        cross_attn = torch.matmul(attn_weights, V)  # [batchsize, miRNA_seq_len, d_model]
        # expand Q mask to mask out queries, zero out padded queries
        valid_counts = Q_mask.sum(dim=1, keepdim=True)  # [batchsize, 1]
        Q_mask = Q_mask.unsqueeze(-1).expand(
            -1, -1, d_model
        )  # [batchsize, miRNA_seq_len, d_model]
        cross_attn = cross_attn * Q_mask # [batchsize, miRNA_seq_len, d_model]
        # average pool over seq_length
        cross_attn = cross_attn.sum(dim=1) / valid_counts  # [batchsize, d_model]
        # print("Cross attention shape = ", cross_attn.shape)
        if return_attn:
            return {"cross_attn": cross_attn, "attn_weights": attn_weights}
        else:
            return {"cross_attn": cross_attn}

    def run_training(
        self,
        model,
        train_loader,
        optimizer,
        epoch,
        loss_fn,
        tokenizer,
        log_interval=10,
        alpha=0.0,
        margin=0.1
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
        })
        config = wandb.config
        # set the entire model to train mode
        model.train()
        epoch_loss=0.0
        loss_ls = []
        optimizer.zero_grad()
        for batch_idx, (
            mRNA_seq,
            miRNA_seq,
            mRNA_seq_mask,
            miRNA_seq_mask,
            seed_start,
            seed_end,
            target,
        ) in enumerate(train_loader):
            mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, seed_start, seed_end, target = (
                mRNA_seq.to(self.device),
                miRNA_seq.to(self.device),
                mRNA_seq_mask.to(self.device),
                miRNA_seq_mask.to(self.device),
                seed_start.to(self.device),
                seed_end.to(self.device),
                target.to(self.device),
            )
            pos_mask = (target == 1)
            pos_indices = torch.where(pos_mask)[0]
            if pos_indices.shape[0] > 0: 
                s_w, s_i, repeats = model.forward(
                    mRNA_seq=mRNA_seq,
                    miRNA_seq=miRNA_seq,
                    mRNA_seq_mask=mRNA_seq_mask,
                    miRNA_seq_mask=miRNA_seq_mask,
                    seed_start=seed_start,
                    seed_end=seed_end,
                    tokenizer=tokenizer,
                    pos_indices=pos_indices,
                    perturb=True,
                )  # (batch_size, 1)
                if s_i is not None:
                    s_w, s_i = (s_w.squeeze(-1), s_i.squeeze(-1))
                    bce_loss = loss_fn(s_w, target.view(-1))
                    s_w_pos = s_w[pos_indices]
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
                else:
                    total_loss = loss_fn(s_w.squeeze(-1), target.view(-1))
                    bce_loss = total_loss.item()
                    wandb.log({
                        "training_bce_loss": bce_loss,
                        "training_total_loss": total_loss.item(),
                    })
            else: # no positive sampels in the batch
                s_w = model.forward(
                    mRNA_seq=mRNA_seq,
                    miRNA_seq=miRNA_seq,
                    mRNA_seq_mask=mRNA_seq_mask,
                    miRNA_seq_mask=miRNA_seq_mask,
                    perturb=False,
                )
                total_loss = loss_fn(s_w.squeeze(-1), target.view(-1))
                
            if self.accumulation_step is not None:
                total_loss = total_loss / self.accumulation_step
                total_loss.backward()
                loss_ls.append(total_loss.item())
                if (batch_idx + 1) % self.accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if self.ddp:
                        print(
                            f"[Rank {dist.get_rank()}] "
                            f"Train Epoch: {epoch} "
                            f"[{(batch_idx + 1) * len(mRNA_seq)}/{len(train_loader.sampler)} "
                            f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)]\t"
                            f"Avg Loss: {sum(loss_ls) / len(loss_ls):.6f}\t", 
                            flush=True
                        )
                    else:
                        print(
                            f"Train Epoch: {epoch} "
                            f"[{(batch_idx + 1) * len(mRNA_seq)}/{len(train_loader.dataset)} "
                            f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)]\t"
                            f"Avg Loss: {sum(loss_ls) / len(loss_ls):.6f}\t", 
                            flush=True
                        )
                    loss_ls = []
            else:
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (batch_idx + 1) % log_interval == 0:
                    if self.ddp:
                        print(
                            f"[Rank {dist.get_rank()}] "
                            f"Train Epoch: {epoch} "
                            f"[{(batch_idx + 1) * len(mRNA_seq)}/{len(train_loader.sampler)} "
                            f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                            f"Loss: {total_loss.item():.6f}\n",
                            flush=True
                        )
                    else:
                        print(
                            f"Train Epoch: {epoch} "
                            f"[{(batch_idx + 1) * len(mRNA_seq)}/{len(train_loader.dataset)} "
                            f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)]\t"
                            f"Loss: {total_loss.item():.6f}\t",
                            flush=True
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
            for mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, seed_start, seed_end, target in test_loader:
                mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, seed_start, seed_end, target = (
                    mRNA_seq.to(self.device),
                    miRNA_seq.to(self.device),
                    mRNA_seq_mask.to(self.device),
                    miRNA_seq_mask.to(self.device),
                    seed_start.to(self.device),
                    seed_end.to(self.device),
                    target.to(self.device),
                )
                pos_mask = (target == 1)
                pos_indices = torch.where(pos_mask)[0]
                if pos_indices.shape[0] > 0:
                    s_w, s_i, repeats = model.forward(
                                mRNA_seq=mRNA_seq,
                                miRNA_seq=miRNA_seq,
                                mRNA_seq_mask=mRNA_seq_mask,
                                miRNA_seq_mask=miRNA_seq_mask,
                                seed_start = seed_start,
                                seed_end = seed_end,
                                tokenizer = tokenizer,
                                pos_indices = pos_indices,
                                perturb = True,
                                ) # (batch_size, 1)
                    s_w, s_i = (s_w.squeeze(-1), s_i.squeeze(-1))
                    bce_loss = loss_fn(s_w, target.view(-1))
                    s_w_pos = s_w[pos_indices] # repeat only the positive samples
                    s_w_pos_repeated = torch.repeat_interleave(s_w_pos, repeats) # (batchsize * seed_len * 3,)
                    difference = s_w_pos_repeated - s_i - margin
                    ranking_loss = loss_fn(difference, torch.ones_like(difference))
                    # ranking_loss = torch.clamp(s_i - s_w_pos_repeated + margin, min=0.0).mean()
                    total_loss = bce_loss + alpha * ranking_loss
                    losses.append(total_loss.item())
                    # Log individual losses to wandb
                    wandb.log({
                        "evaluation_bce_loss": bce_loss.item(),
                        "evaluation_ranking_loss": ranking_loss.item(),
                        "evaluation_total_loss": total_loss.item(),
                    })
                else: 
                    s_w = model.forward(mRNA_seq=mRNA_seq,
                                        miRNA_seq=miRNA_seq,
                                        mRNA_seq_mask=mRNA_seq_mask,
                                        miRNA_seq_mask=miRNA_seq_mask,
                                        perturb=False,
                                        )
                    s_w = s_w.squeeze(-1)
                    total_loss = loss_fn(s_w, target.view(-1)) # total loss equals bce loss when no perturbation
                    losses.append(total_loss.item())
                predictions = (s_w > 0.5).long()
                if self.ddp:
                    local_correct += predictions.eq(target.view(-1)).sum().item()
                else:
                    correct += predictions.eq(target.view(-1)).sum().item()
                unperturbed_predictions.extend(s_w_pos_repeated.cpu())
                perturbed_predictions.extend(s_i.cpu())
            # change in prediction
            mean_unperturbed_score = np.mean(np.asarray(unperturbed_predictions))
            mean_perturbed_score = np.mean(np.asarray(perturbed_predictions))
            diff_score = (mean_unperturbed_score - mean_perturbed_score).item()
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
            print(
                f"\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} "
                f"({accuracy:2f})"
            )
            avg_loss = sum(losses) / len(losses)
            print(f"Average unperturbed prediction: {mean_unperturbed_score:.4f}")
            print(f"Average perturbed prediction: {mean_perturbed_score:.4f}")
            print(f"Difference (unperturbed - perturbed): {diff_score:.4f}")
            # Log individual losses to wandb
            wandb.log({
                "Accuracy": accuracy,
                "Evaluation epoch loss": avg_loss, 
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
        """Test loop."""
        model.eval()
        
        predictions = []
        true_labels = []
        with torch.no_grad():
            for mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, seed_start, seed_end, target in test_loader:
                mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target = (
                    mRNA_seq.to(self.device),
                    miRNA_seq.to(self.device),
                    mRNA_seq_mask.to(self.device),
                    miRNA_seq_mask.to(self.device),
                    seed_start.to(self.device),
                    seed_end.to(self.device),
                    target.to(self.device),
                )
                output = model.forward(
                            mRNA_seq=mRNA_seq,
                            miRNA_seq=miRNA_seq,
                            mRNA_seq_mask=mRNA_seq_mask,
                            miRNA_seq_mask=miRNA_seq_mask,
                            perturb=False,
                            )  # (batch_size, 1)
                probabilities = torch.sigmoid(output.squeeze(-1)).cpu().numpy().tolist()
                targets = target.cpu().view(-1).numpy().tolist()
                predictions.extend(probabilities)
                true_labels.extend(targets)
        
        acc = self.assess_acc(predictions=predictions, targets=true_labels)
        return acc, predictions, true_labels
    
    def forward(self, 
                mRNA_seq,
                miRNA_seq,
                mRNA_seq_mask=None,
                seed_start:torch.tensor=-1,
                seed_end:torch.tensor=-1,
                miRNA_seq_mask=None,
                return_attn=False,
                perturb=False,
                pos_indices:torch.tensor=None,
                tokenizer=None
                ):
        """
        Forward pass for the model.

        Args:
            mRNA_seq (torch.Tensor): mRNA sequence tensor of shape (batch_size, mRNA_seq_len).
            miRNA_seq (torch.Tensor): miRNA sequence tensor of shape (batch_size, miRNA_seq_len).
            mRNA_seq_mask (torch.Tensor): Mask for mRNA sequence of shape (batch_size, mRNA_seq_len).
            miRNA_seq_mask (torch.Tensor): Mask for miRNA sequence of shape (batch_size, miRNA_seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_classes).
        """
        # Extract mRNA and miRNA hidden states
        mRNA_hidden_states = self.hyena(
            input_ids=mRNA_seq,
            use_only_miRNA=False,
            max_mRNA_length=self.mRNA_max_len,
            max_miRNA_length=self.miRNA_max_len
        )  # (batch_size, mRNA_seq_len, d_model)
        miRNA_hidden_states = self.hyena(
            input_ids=miRNA_seq, 
            use_only_miRNA=False,
            max_mRNA_length=self.mRNA_max_len,
            max_miRNA_length=self.miRNA_max_len
        )  # (batch_size, miRNA_seq_len, d_model)
        
        # Compute Q, K, V for cross-attention
        Q = self.q_layer(miRNA_hidden_states)  # (batch_size, miRNA_seq_len, d_model)
        K = self.kv_layer(mRNA_hidden_states)  # (batch_size, mRNA_seq_len, d_model)
        V = self.kv_layer(mRNA_hidden_states)  # (batch_size, mRNA_seq_len, d_model)

        # Compute cross-attention
        output = self.compute_cross_attention(
            Q=Q,
            K=K,
            V=V,
            Q_mask=miRNA_seq_mask,
            K_mask=mRNA_seq_mask,
            return_attn=return_attn,
        )  # (batch_size, d_model)

        if return_attn:
            attn_weights=output["attn_weights"]
            cross_attn=output["cross_attn"]
        else:
            cross_attn=output["cross_attn"]
        
        # Pass through the MLP head
        s_w = self.mlp_head(cross_attn)  # (batch_size, n_classes)
        
        if perturb:
            assert (pos_indices is not None) and (pos_indices.shape[0]>0), "Only Positive samples will be perturbed during training. Positive samples are not provided."
            assert tokenizer is not None, "Tokenizer to tokenize perturbed sequences. Tokenizer can not be None."
            
            perturbed_seqs = []
            perturbed_masks = []
            repeats = [] # the number of time to repeat s_w
            for idx in pos_indices:
                original_seq = mRNA_seq[idx]
                original_mask = mRNA_seq_mask[idx]
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
            if len(perturbed_seqs) > 1:
                repeats = torch.tensor(repeats, dtype=torch.long, device=self.device)
                # perturbed_seqs has shape (batchsize * seed_len * 3)
                # Compute perturbed scores (s_i)
                perturbed_mRNA_hidden_states = self.hyena(
                    input_ids = torch.stack(perturbed_seqs).to(self.device),
                    input_mask = torch.stack(perturbed_masks).to(self.device),
                    max_mRNA_length=self.mRNA_max_len,
                    max_miRNA_length=self.miRNA_max_len,
                    ) # (batchsize * seed_len * 3, mRNA_seq_len, hidden_size)
                miRNA_hidden_states_pos = miRNA_hidden_states[pos_indices]
                miRNA_hidden_states_pos = torch.repeat_interleave(miRNA_hidden_states_pos, repeats, dim=0)
                # Compute Q, K, V for cross-attention
                # new_batchsize = batchsize * seed_len * 3
                Q = self.q_layer(miRNA_hidden_states_pos)  # (new_batchsize, miRNA_seq_len, hidden_size)
                K = self.kv_layer(perturbed_mRNA_hidden_states)  # (new_batchsize, mRNA_seq_len, hidden_size)
                V = self.kv_layer(perturbed_mRNA_hidden_states)  # (new_batchsize, mRNA_seq_len, hidden_size)
                Q_mask = torch.repeat_interleave(miRNA_seq_mask[pos_indices], repeats, dim=0) # (new_batchsize, mRNA_seq_len)
                K_mask = torch.stack(perturbed_masks).to(self.device)
                # Compute cross-attention
                output = self.compute_cross_attention(
                    Q=Q,
                    K=K,
                    V=V,
                    Q_mask=Q_mask,
                    K_mask=K_mask,
                    return_attn=return_attn,
                )  # (new_batchsize, d_model)

                if return_attn:
                    attn_weights=output["attn_weights"]
                    cross_attn=output["cross_attn"]
                else:
                    cross_attn=output["cross_attn"]
                
                # Pass through the MLP head
                s_i = self.mlp_head(cross_attn)  # (new_batchsize, n_classes)
                
                if return_attn:
                    return s_w, s_i, repeats, attn_weights
                else:
                    return s_w, s_i, repeats
            else:
                return s_w, None, repeats
        
        if return_attn:
            return s_w, attn_weights
        else:
            return s_w        
