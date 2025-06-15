import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import math
import wandb
import random
import numpy as np
from time import time

from utils import load_dataset
from Data_pipeline import CharacterTokenizer, QuestionAnswerDataset

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device='cuda'):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, 
                query, 
                key, 
                value, 
                mask=None):
        batch_size = query.size(0)
        len_q, len_k, len_v = query.size(1), key.size(1), value.size(1)
        
        # Linear transformations and split into heads
        # [batchsize, seq_len, (num_heads*head_dim)]
        Q = self.query(query).view(batch_size, len_q, self.num_heads, self.head_dim) 
        K = self.key(key).view(batch_size, len_k, self.num_heads, self.head_dim)
        V = self.value(value).view(batch_size, len_v, self.num_heads, self.head_dim)

        # Transpose 
        # [batchsize, num_heads, seq_len, head_dim]
        Q = Q.transpose(1,2) 
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(2, 3)) * self.scale # (batchsize, num_head, q_len, k_len)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, Q.shape[2], -1) # (batchsize, head_dim, q_len, k_len)
            mask = mask.to(self.device)
            scores = scores.masked_fill(mask==0, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, p=0.1)
        output = torch.matmul(attention, V) # [batchsize, num_heads, q_len, head_dim]

        # Concatenate heads and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim) # [batchsize, q_len, embed_dim]
        output = self.out(output) # [batchsize, q_len, embed_dim]

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class AdditivePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # Learnable positional embedding: shape [max_len, d_model]
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        # Position indices: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)  # [batch_size, seq_len]

        # print("position_ids.max() =", position_ids.max().item())
        # print("embedding size =", self.pos_embedding.num_embeddings)
        
        # Get positional embeddings
        pos_emb = self.pos_embedding(position_ids)  # [batch_size, seq_len, d_model]
        return x + pos_emb

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, device='cuda'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, device)
        self.feed_forward = PositionWiseFeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-Head Attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Position-wise Feed-Forward Network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1, device='cuda'):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim=embed_dim, 
                                     num_heads=num_heads, 
                                     ff_dim=ff_dim, 
                                     dropout=dropout, 
                                     device=device) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class LinearHead(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_sizes,
                 output_size,
                 dropout):
        super(LinearHead, self).__init__()
        self.activation = nn.ReLU()
        layers = []
        for h in hidden_sizes:
            layer = nn.Linear(input_size, h)
            layers += [
                layer,
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            input_size = h
        layers.append(nn.Linear(h, output_size))
        self.transform = nn.Sequential(*layers)
    def forward(self, x):
        return self.transform(x)

class CrossAttentionPredictor(nn.Module):
    def __init__(self,  
                 mirna_max_len:int,
                 mrna_max_len:int, 
                 vocab_size:int=12, # 7 special tokens + 5 bases
                 num_layers:int=2, 
                 embed_dim:int=256, 
                 num_heads:int=2, 
                 ff_dim:int=512,
                 hidden_sizes:list[int]=[512, 512],
                 n_classes:int=1, 
                 dropout_rate:float=0.1,
                 device:str='cuda',
                 predict_span=True,
                 predict_binding=False):
        super(CrossAttentionPredictor, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mirna_positional_embedding = AdditivePositionalEncoding(max_len=mirna_max_len, d_model=embed_dim)
        self.mrna_positional_embedding = AdditivePositionalEncoding(max_len=mrna_max_len, d_model=embed_dim)
        self.mirna_encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            device=device,
            dropout=dropout_rate,
        )
        self.mrna_encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            device=device,
            dropout=dropout_rate,
        )
        self.cross_attn_layer = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.cross_norm = nn.LayerNorm(embed_dim) # normalize over embedding dimension
        self.qa_outputs = nn.Linear(embed_dim, 2) # Linear head instead of one Linear transformation
        self.binding_output = LinearHead(
            input_size=embed_dim, 
            hidden_sizes=hidden_sizes,
            output_size=n_classes,
            dropout=dropout_rate)
        self.predict_span = predict_span
        self.predict_binding = predict_binding
    
    def forward(self, mirna, mrna, mrna_mask, mirna_mask):
        mirna_embedding = self.embedding(mirna)
        mirna_embedding = self.mirna_positional_embedding(mirna_embedding) # (batch_size, mirna_len, embed_dim)
        mrna_embedding = self.embedding(mrna)
        mrna_embedding = self.mrna_positional_embedding(mrna_embedding) # (batch_size, mrna_len, embed_dim)
        
        mirna_embedding = self.mirna_encoder(mirna_embedding, mask=mirna_mask)  # (batch_size, mirna_len, embed_dim)
        mrna_embedding = self.mrna_encoder(mrna_embedding, mask=mrna_mask) # (batch_size, mrna_len, embed_dim)
        
        z = self.cross_attn_layer(query=mrna_embedding, 
                                  key=mirna_embedding,
                                  value=mirna_embedding,
                                  mask=mirna_mask) # pass key-mask
        z_res = self.dropout(z) + mrna_embedding
        z_norm = self.cross_norm(z_res)
        # 消除padded tokens 在隐藏状态中的信息
        z_norm = z_norm.masked_fill(mrna_mask.unsqueeze(-1)==0, 0) # (batch_size, mrna_len, embed_dim)
        
        if self.predict_binding:
            valid_counts = mrna_mask.sum(dim=1, keepdim=True) # (batch_size)
            # avg pooling over seq_len
            z_norm_mean = z_norm.sum(dim=1) / (valid_counts + 1e-8) # (batch_size, embed_dim)
            # predict binding
            binding_logits = self.binding_output(z_norm_mean) # (batchsize, 1)
        else:
            binding_logits = None
        if self.predict_span:
            # predict start and end
            span_logits = self.qa_outputs(z_norm) # (batchsize, mrna_len, 2)
            start_logits, end_logits = span_logits[...,0], span_logits[...,1] # (batchsize, mrna_len)
        else:
            start_logits, end_logits = None, None
        return binding_logits, start_logits, end_logits
    
class QuestionAnsweringModel(nn.Module):
    def __init__(self,
                mrna_max_len,
                mirna_max_len,
                train_datapath,
                valid_datapath,
                device: str='cuda',
                epochs:int=100,
                batch_size:int=64,
                lr=0.001,
                seed=42,
                predict_span=True,
                predict_binding=False,
                use_cross_attn=True,):
        super(QuestionAnsweringModel, self).__init__()
        self.mrna_max_len = mrna_max_len
        self.mirna_max_len = mirna_max_len
        self.train_datapath = train_datapath
        self.valid_datapath = valid_datapath
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.predict_binding = predict_binding
        self.predict_span = predict_span
        if use_cross_attn:
            self.predictor = CrossAttentionPredictor(mrna_max_len=mrna_max_len,
                                                    mirna_max_len=mirna_max_len,
                                                    device=device,
                                                    predict_span=predict_span,
                                                    predict_binding=predict_binding)
    
    def forward(self, 
                mirna, 
                mrna, 
                mrna_mask, 
                mirna_mask,):
        return self.predictor(mirna=mirna, 
                              mrna=mrna, 
                              mrna_mask=mrna_mask,
                              mirna_mask=mirna_mask,)
    
    @staticmethod
    def compute_span_metrics(start_preds, end_preds, start_labels, end_labels):
        """
        计算 exact match 和 F1 score.
        输入张量都是 [B]，代表每个样本的 start/end.
        """
        exact_matches = 0
        f1_total = 0.0
        n = len(start_preds)

        for i in range(n):
            pred_start = int(start_preds[i])
            pred_end   = int(end_preds[i])
            true_start = int(start_labels[i])
            true_end   = int(end_labels[i])

            # Compute overlap
            overlap_start = max(pred_start, true_start)
            overlap_end   = min(pred_end, true_end)
            overlap       = max(0, overlap_end - overlap_start)

            pred_len = max(1, pred_end - pred_start)
            true_len = max(1, true_end - true_start)

            precision = overlap / pred_len
            recall = overlap / true_len
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            f1_total += f1

            # Exact match
            if pred_start == true_start and pred_end == true_end:
                exact_matches += 1

        return {
            "exact_match": exact_matches / n,
            "f1": f1_total / n,
        }
    
    def train_loop(self, 
              model, 
              dataloader, 
              loss_fn,
              optimizer, 
              device,
              epoch,
              accumulation_step=1):
        '''
        Training loop
        '''
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        loss_list = []
        for batch_idx, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)

            mirna_mask = batch["mirna_attention_mask"]
            mrna_mask  = batch["mrna_attention_mask"]
            
            outputs = model(
                mirna=batch["mirna_input_ids"],
                mrna=batch["mrna_input_ids"],
                mirna_mask=mirna_mask,
                mrna_mask=mrna_mask,
            )
            binding_logits, start_logits, end_logits = outputs  # [B, L]
               
            if self.predict_span:
                # mask padded output in start and end logits
                start_logits = start_logits.masked_fill(mrna_mask==0, float("-inf"))
                end_logits   = end_logits.masked_fill(mrna_mask==0, float("-inf"))
            
            start_positions = batch["start_positions"]
            end_positions   = batch["end_positions"]
            binding_targets = batch["target"]

            span_loss = torch.tensor(0.0, device=device)
            binding_loss = torch.tensor(0.0, device=device)

            if self.predict_binding and binding_logits is not None:
                binding_loss_fn = nn.BCEWithLogitsLoss()
                binding_loss    = binding_loss_fn(binding_logits.squeeze(-1), binding_targets.view(-1).float())
                pos_mask        = binding_targets.view(-1).bool()
                if self.predict_span and pos_mask.any():
                    # only loss of positive pairs are counted
                    loss_start = loss_fn(start_logits[pos_mask,], start_positions[pos_mask]) # CrossEntropyLoss expects [B, L], labels as [B]
                    loss_end   = loss_fn(end_logits[pos_mask,], end_positions[pos_mask])
                    span_loss  = 0.5 * (loss_start + loss_end)
                    loss       = span_loss + binding_loss
                else:
                    loss       = binding_loss
            elif self.predict_span:
                # assume all mirna-mrna pairs are positive
                # CrossEntropyLoss expects [B, L], labels as [B]
                loss_start = loss_fn(start_logits, start_positions)
                loss_end   = loss_fn(end_logits, end_positions)
                span_loss  = 0.5 * (loss_start + loss_end)
                loss       = span_loss 

            loss = loss / accumulation_step
            loss.backward()
            bs = batch["mrna_input_ids"].size(0)
            if accumulation_step != 1:
                loss_list.append(loss.item())
                if (batch_idx + 1) % accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print(
                        f"Train Epoch: {epoch} "
                        f"[{(batch_idx + 1) * bs}/{len(dataloader.dataset)} "
                        f"({(batch_idx + 1) * bs / len(dataloader.dataset) * 100:.0f}%)] "
                        f"Avg loss: {sum(loss_list) / len(loss_list):.6f}\n",
                        flush=True
                    )
                    loss_list = []
            else:
                optimizer.step()
                optimizer.zero_grad()
                print(
                    f"Train Epoch: {epoch} "
                    f"[{(batch_idx + 1) * bs}/{len(dataloader.dataset)} "
                    f"({(batch_idx + 1) * bs/len(dataloader.dataset) * 100:.0f}%)] "
                    f"Span Loss: {span_loss.item():.6f} "
                    f"Binding Loss: {binding_loss.item():.6f}\n",
                    flush=True
                ) 

            total_loss += loss.item() * accumulation_step
            wandb.log({
                "train_cross_entropy_loss": loss.item()
            })
        # After the loop, if gradients remain (for non-divisible number of batches)
        if (batch_idx + 1) % self.accumulation_step != 0:
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        wandb.log({
            "train_epoch_loss": avg_loss,
        })
        return avg_loss

    def eval_loop(self, model, dataloader, device):
        model.eval()
        total_loss        = 0.0
        all_start_preds   = []
        all_end_preds     = []
        all_binding_preds = []
        all_start_labels  = []
        all_end_labels    = []
        all_binding_labels = []

        with torch.no_grad():
            for batch in dataloader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                mrna_mask  = batch["mrna_attention_mask"]
                mirna_mask = batch["mirna_attention_mask"]
                outputs    = model(
                    mirna=batch["mirna_input_ids"],
                    mrna=batch["mrna_input_ids"],
                    mirna_mask=mirna_mask,
                    mrna_mask=mrna_mask,
                )
                binding_logits, start_logits, end_logits = outputs

                if self.predict_span:
                    # mask padded mrna tokens
                    start_logits = start_logits.masked_fill(mrna_mask==0, float("-inf"))
                    end_logits   = end_logits.masked_fill(mrna_mask==0, float("-inf"))
                    start_positions = batch["start_positions"] # (batchsize, )
                    end_positions   = batch["end_positions"] # (batchsize, )

                # Compute loss
                loss_fn = nn.CrossEntropyLoss()

                # Compute binding loss if binding label is predicted
                if self.predict_binding and binding_logits is not None:
                    binding_targets = batch["target"] # (batchsize, )
                    binding_loss_fn = nn.BCEWithLogitsLoss()
                    binding_loss    = binding_loss_fn(binding_logits.squeeze(-1), binding_targets.view(-1).float())
                    binding_probs   = F.sigmoid(binding_logits)
                    binding_preds   = (binding_probs > 0.5).to(torch.int)
                    all_binding_preds.extend(binding_preds.cpu())
                    all_binding_labels.extend(binding_targets.view(-1).cpu())
                    pos_mask = binding_targets.view(-1).bool() # (batchsize, )
                    if self.predict_span and pos_mask.any():
                        loss_start = loss_fn(start_logits[pos_mask,], start_positions[pos_mask])
                        loss_end = loss_fn(end_logits[pos_mask,], end_positions[pos_mask])
                        span_loss = 0.5 * (loss_start + loss_end)
                        loss = span_loss + binding_loss
                    else:
                        loss = binding_loss
                elif self.predict_span:
                    # assume all mirna-mrna pairs are positive
                    loss_start = loss_fn(start_logits, start_positions)
                    loss_end   = loss_fn(end_logits, end_positions)
                    span_loss  = 0.5 * (loss_start + loss_end)
                    loss       = span_loss 
                
                total_loss += loss.item()
                wandb.log({
                    "val_cross_entropy_loss": loss.item()
                })

                # Predictions
                pos_mask = binding_targets.view(-1).bool()
                if self.predict_span and pos_mask.any():
                    start_preds = torch.argmax(start_logits[pos_mask,], dim=-1) #(batch_size, )
                    end_preds   = torch.argmax(end_logits[pos_mask,], dim=-1) #(batch_size, )

                    all_start_preds.extend(start_preds.cpu())
                    all_end_preds.extend(end_preds.cpu())
                    all_start_labels.extend(start_positions[pos_mask].cpu())
                    all_end_labels.extend(end_positions[pos_mask].cpu())

        # if there are positive examples
        if len(all_start_preds) > 0:
            all_start_preds  = torch.stack(all_start_preds).detach().cpu().long()
            all_start_labels = torch.stack(all_start_labels).detach().cpu().long()
            all_end_preds    = torch.stack(all_end_preds).detach().cpu().long()
            all_end_labels   = torch.stack(all_end_labels).detach().cpu().long()
            acc_start        = (all_start_preds == all_start_labels).float().mean().item()
            acc_end          = (all_end_preds == all_end_labels).float().mean().item()
            span_metrics     = self.compute_span_metrics(
                all_start_preds, all_end_preds, all_start_labels, all_end_labels)
            exact_match      = span_metrics["exact_match"]
            f1               = span_metrics["f1"]
        else:
            print("No positive example in this epoch. No span metrics is measured.")
            acc_start   = 0.0
            acc_end     = 0.0
            exact_match = 0.0
            f1          = 0.0

        if self.predict_binding:
            all_binding_labels = torch.tensor(all_binding_labels, dtype=torch.long)
            all_binding_preds  = torch.tensor(all_binding_preds, dtype=torch.long)
            acc_binding        = (all_binding_preds == all_binding_labels).float().mean().item()
        else:
            acc_binding        = 0.0

        avg_loss = total_loss / len(dataloader)
        
        print(f"Start Acc:   {acc_start*100}%\n"
              f"End Acc:     {acc_end*100}%\n"
              f"Span Exact Match: {exact_match*100}%\n"
              f"F1 Score:    {f1}\n"
              f"Binding Acc: {acc_binding*100}")
        
        wandb.log({
            "val_epoch_loss":  avg_loss,
            "val_start_acc":   acc_start,
            "val_end_acc":     acc_end,
            "val_exact_match": exact_match,
            "val_f1":          f1,
            "val_binding_acc": acc_binding
        })

        return acc_binding, acc_start, acc_end, exact_match, f1
    
    @staticmethod 
    def seed_everything(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    def run(self, 
            model,
            accumulation_step=1):
        # weights and bias initialization
        wandb.login(key="600e5cca820a9fbb7580d052801b3acfd5c92da2")
        wandb.init(
            project="mirna-Question-Answering",
            name=f"binding-random-start-len:{self.mrna_max_len}-epoch:{self.epochs}-batchsize:{self.batch_size}-2layerTrans-512MLP_hidden", 
            config={
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning rate": self.lr,
            }
        )
        self.seed_everything(seed=self.seed)
        tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                       add_special_tokens=False, 
                                       model_max_length=self.mrna_max_len,
                                       padding_side="right")
        # load dataset
        D_train  = load_dataset(self.train_datapath, sep=',')
        D_val    = load_dataset(self.valid_datapath, sep=',')
        ds_train = QuestionAnswerDataset(data=D_train,
                                         mrna_max_len=self.mrna_max_len,
                                         mirna_max_len=self.mirna_max_len,
                                         tokenizer=tokenizer,
                                         seed_start_col="seed start",
                                         seed_end_col="seed end",)
        ds_val = QuestionAnswerDataset(data=D_val,
                                       mrna_max_len=self.mrna_max_len,
                                       mirna_max_len=self.mirna_max_len,
                                       tokenizer=tokenizer,
                                       seed_start_col="seed start",
                                       seed_end_col="seed end",)
        train_loader = DataLoader(ds_train, 
                            batch_size=self.batch_size, 
                            shuffle=True)
        val_loader   = DataLoader(ds_val, 
                                batch_size=self.batch_size, 
                                shuffle=False)
        loss_fn   = nn.CrossEntropyLoss()
        model.to(self.device)
        
        if self.predict_binding and not self.predict_span:
            # freeze update of params in the span prediction head
            for p in model.predictor.qa_outputs.parameters():
                p.requires_grad = False
        elif self.predict_span and not self.predict_binding:
            # freeze update of params in the binding prediction head
            for p in model.predictor.binding_output.parameters():
                p.requires_grad = False

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=1e-2)

        start    = time()
        best_binding_acc = 0
        count    = 0
        patience = 10
        model_checkpoints_dir = os.path.join(
            PROJ_HOME, 
            "checkpoints", 
            "TargetScan", 
            "TwoTowerTransformer", 
            str(self.mRNA_max_len),
        )
        os.makedirs(model_checkpoints_dir, exist_ok=True)
        for epoch in range(self.epochs):
            self.train_loop(model=model,
                       dataloader=train_loader,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       device=self.device,
                       epoch=epoch,
                       accumulation_step=accumulation_step)
            acc_binding, acc_start, acc_end, exact_match, f1 = self.eval_loop(model=model,
                                                                            dataloader=val_loader,
                                                                            device=self.device,)
            if acc_binding >= best_binding_acc:
                best_binding_acc = acc_binding
                count = 0
                ckpt_name = f"best_binding_acc_{acc_binding:.4f}_epoch{epoch}.pth"
                torch.save(model.state_dict(), os.path.join(model_checkpoints_dir, ckpt_name))
                wandb.save(ckpt_name)
            else:
                count += 1
                if count == patience:
                    print("Max patience reached with no improvement on accuracy. Early stopped training.")
                    break
            cost = time() - start
            remain = cost/(epoch + 1) * (self.epochs - epoch - 1) /3600
            print(f'still remain: {remain} hrs.')
        wandb.run.summary["best_binding_acc"] = best_binding_acc

if __name__ == "__main__":
    torch.cuda.empty_cache() # clear crashed cache
    mrna_max_len = 30
    mirna_max_len = 24
    PROJ_HOME = os.path.expanduser("~/projects/mirLM")
    train_datapath = os.path.join(PROJ_HOME, "TargetScan_dataset/TargetScan_train_30_randomized_start.csv")
    valid_datapath = os.path.join(PROJ_HOME, "TargetScan_dataset/TargetScan_validation_30_randomized_start.csv")
    
    model = QuestionAnsweringModel(mrna_max_len=mrna_max_len,
                                   mirna_max_len=mirna_max_len,
                                   train_datapath=train_datapath,
                                   valid_datapath=valid_datapath,
                                   device='cuda:1',
                                   epochs=100,
                                   batch_size=32,
                                   lr=1e-4,
                                   seed=54,
                                   predict_span=False,
                                   predict_binding=True)
    model.run(model=model,
              accumulation_step=8)
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters = {n_params}.") #11.3M
   
