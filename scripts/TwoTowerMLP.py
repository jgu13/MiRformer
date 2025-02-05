import os
import math
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
# Local imports
from mirLM import mirLM
from Hyena_layer import LinearHead

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

class TwoTowerMLP(mirLM):
    def __init__(
        self,
        mRNA_max_len: int,
        miRNA_max_len: int,
        train_dataset_path: str,
        val_dataset_path: str,
        test_dataset_path: str,
        dataset_name: str,
        model_name: str,
        device: str,
        epochs: int=10,
        batch_size: int=256,
        ddp_flag: bool=False,
        resume_ckpt: str=None,
        evaluate_flag: bool=False,        
        pretrained_model_name:str=None,
        backbone_cfg=None,
        # exps params
        n_classes=None,
        use_padding:bool=True,
        use_head=False,
        rc_aug:bool=False,
        add_eos:bool=False,
        download=False,
        hidden_sizes:int=None,
    ):
        """
        Args:
            pretrained_model_name (str): Name of the pretrained model.
            backbone_cfg (dict): Configuration for the backbone model.
            n_classes (int): Number of output classes.
            use_head (bool): Whether to use the head of the pretrained model.
            device (str): Device to use (e.g., "cuda" or "cpu").
            download (bool): Whether to download the pretrained model.
            hidden_sizes (list): Hidden sizes for the MLP head.
        """
        super(TwoTowerMLP, self).__init__(
            mRNA_max_length=mRNA_max_len,
            miRNA_max_length=miRNA_max_len,
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            test_dataset_path=test_dataset_path,
            dataset_name=dataset_name,
            model_name=model_name,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            ddp_flag=ddp_flag,
            resume_ckpt=resume_ckpt,
            evaluate_flag=evaluate_flag,
            # exps params
            use_padding=use_padding,
            use_head=use_head,
            rc_aug=rc_aug,
            add_epos=add_eos,
            download=download,
            hidden_sizes=hidden_sizes,
        )

        # Initialize the MLP head
        if hidden_sizes is None:
            hidden_sizes = [backbone_cfg["d_model"] * 2, backbone_cfg["d_model"] * 2]
        self.mlp_head = LinearHead(
            d_model=backbone_cfg["d_model"], 
            d_output=n_classes, 
            hidden_sizes=hidden_sizes,
        )

        # Initialize Q and KV layers
        self.q_layer = nn.Linear(backbone_cfg["d_model"], backbone_cfg["d_model"])
        self.kv_layer = nn.Linear(backbone_cfg["d_model"], backbone_cfg["d_model"])
     
    def compute_cross_attention(
            self,
            Q: torch.tensor, 
            K: torch.tensor, 
            V: torch.tensor, 
            Q_mask: torch.tensor, 
            K_mask: torch.tensor
        ):
        '''
        Compute cross attention
        '''
        d_model = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)  # [batchsize, miRNA_seq_len, mRNA_seq_len]
        # expand K mask to mask out keys, make each query only attend to valid keys
        K_mask = K_mask.unsqueeze(1).expand(
            -1, Q.shape[1], -1
        )  # [batchsize, miRNA_seq_len, mRNA_seq_len]
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
        return cross_attn

    def train(
        self,
        train_loader,
        optimizer,
        epoch,
        loss_fn,
        log_interval=10,
        epoch_loss=0.0,
    ):
        """Training loop."""
        self.MLP_head.eval()
        self.HyenaDNA_feature_extractor.eval()
        
        loss_ls = []
        for batch_idx, (
            mRNA_seq,
            miRNA_seq,
            mRNA_seq_mask,
            miRNA_seq_mask,
            target,
        ) in enumerate(train_loader):
            mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target = (
                mRNA_seq.to(self.device),
                miRNA_seq.to(self.device),
                mRNA_seq_mask.to(self.device),
                miRNA_seq_mask.to(self.device),
                target.to(self.device),
            )
            output = self.forward(                
                            device=self.device,
                            mRNA_seq=mRNA_seq,
                            miRNA_seq=miRNA_seq,
                            mRNA_seq_mask=mRNA_seq_mask,
                            miRNA_seq_mask=miRNA_seq_mask
                            )  # (batch_size, 1)
            loss = loss_fn(output.squeeze().sigmoid(), target.squeeze())

            if self.accumulation_step is not None:
                loss = loss / self.accumulation_step
                loss_ls.append(loss.item())
                loss.backward()
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
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (batch_idx + 1) % log_interval == 0:
                    if self.ddp:
                        print(
                            f"[Rank {dist.get_rank()}] "
                            f"Train Epoch: {epoch} "
                            f"[{(batch_idx + 1) * len(mRNA_seq)}/{len(train_loader.sampler)} "
                            f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)] "
                            f"Loss: {loss.item():.6f}\n",
                            flush=True
                        )
                    else:
                        print(
                            f"Train Epoch: {epoch} "
                            f"[{(batch_idx + 1) * len(mRNA_seq)}/{len(train_loader.dataset)} "
                            f"({100.0 * (batch_idx + 1) / len(train_loader):.0f}%)]\t"
                            f"Loss: {loss.item():.6f}\t",
                            flush=True
                        )
            epoch_loss += loss.item()
        average_loss = epoch_loss / len(train_loader)
        return average_loss
    
    def test(
         self,
         test_loader):
        """Test loop."""
        self.MLP_head.eval()
        self.HyenaDNA_feature_extractor.eval()
        if self.ddp:
            local_correct = 0
        else:
            correct = 0
        with torch.no_grad():
            for mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target in test_loader:
                mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target = (
                    mRNA_seq.to(self.device),
                    miRNA_seq.to(self.device),
                    mRNA_seq_mask.to(self.device),
                    miRNA_seq_mask.to(self.device),
                    target.to(self.device),
                )
                output = self.forward(                
                            device=self.device,
                            mRNA_seq=mRNA_seq,
                            miRNA_seq=miRNA_seq,
                            mRNA_seq_mask=mRNA_seq_mask,
                            miRNA_seq_mask=miRNA_seq_mask
                            )  # (batch_size, 1)
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
            accuracy = 100.0 * correct / len(test_loader.dataset)
            print(
                f"\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} "
                f"({accuracy:2f})"
            )
            return accuracy       

    def evaluate(
            self,
            test_loader):
        """Test loop."""
        self.MLP_head.eval()
        self.HyenaDNA_feature_extractor.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target in test_loader:
                mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target = (
                    mRNA_seq.to(device),
                    miRNA_seq.to(device),
                    mRNA_seq_mask.to(device),
                    miRNA_seq_mask.to(device),
                    target.to(device),
                )
                output = self.forward(                
                            device=self.device,
                            mRNA_seq=mRNA_seq,
                            miRNA_seq=miRNA_seq,
                            mRNA_seq_mask=mRNA_seq_mask,
                            miRNA_seq_mask=miRNA_seq_mask
                            )  # (batch_size, 1)
                probabilities = torch.sigmoid(output.squeeze()).cpu().numpy().tolist()
                targets = target.cpu().view(-1).numpy().tolist()
                predictions.extend(probabilities)
                true_labels.extend(targets)

        return predictions, true_labels
    
    def forward(self, 
                device,
                mRNA_seq,
                miRNA_seq,
                mRNA_seq_mask=None,
                miRNA_seq_mask=None
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
            device=device,
            input_ids=mRNA_seq,
            use_only_miRNA=False
        )  # (batch_size, mRNA_seq_len, d_model)
        miRNA_hidden_states = self.hyena(
            device=device,
            input_ids=miRNA_seq, 
            use_only_miRNA=False
        )  # (batch_size, miRNA_seq_len, d_model)

        # Compute Q, K, V for cross-attention
        Q = self.q_layer(miRNA_hidden_states)  # (batch_size, miRNA_seq_len, d_model)
        K = self.kv_layer(mRNA_hidden_states)  # (batch_size, mRNA_seq_len, d_model)
        V = self.kv_layer(mRNA_hidden_states)  # (batch_size, mRNA_seq_len, d_model)

        # Compute cross-attention
        cross_attn_output = self.compute_cross_attention(
            Q=Q,
            K=K,
            V=V,
            Q_mask=miRNA_seq_mask,
            K_mask=mRNA_seq_mask,
        )  # (batch_size, d_model)

        # Pass through the MLP head
        output = self.mlp_head(cross_attn_output)  # (batch_size, n_classes)

        return output        
