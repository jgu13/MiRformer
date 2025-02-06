import os
import math
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
# Local imports
from Hyena_layer import HyenaDNAPreTrainedModel, HyenaDNAModel, LinearHead

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

class TwoTowerMLP(nn.Module):
    def __init__(
        self,
        pretrained_model_name,
        device,
        backbone_cfg=None,
        n_classes=1,
        use_head=False,
        download=False,
        hidden_sizes=None,
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
        super(TwoTowerMLP, self).__init__()
        
        # instantiate the model (pretrained here)
        if pretrained_model_name in [
            "hyenadna-tiny-1k-seqlen",
            "hyenadna-small-32k-seqlen",
            "hyenadna-medium-160k-seqlen",
            "hyenadna-medium-450k-seqlen",
            "hyenadna-large-1m-seqlen",
        ]:
            path = f"{PROJ_HOME}/checkpoints"
            # use the pretrained Huggingface wrapper instead
            self.hyena = HyenaDNAPreTrainedModel.from_pretrained(
                                        path,
                                        pretrained_model_name,
                                        download=download,
                                        config=backbone_cfg,
                                        device=device,
                                        use_head=use_head,
                                        n_classes=n_classes,
                                    )
            pretrained_model_name_or_path = os.path.join(path, pretrained_model_name)
            if os.path.isdir(pretrained_model_name_or_path):
                if backbone_cfg is None:
                    with open(
                        os.path.join(pretrained_model_name_or_path, "config.json"),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        backbone_cfg = json.load(f)
                elif isinstance(backbone_cfg, str) and backbone_cfg.endswith(".json"):
                    with open(backbone_cfg, "r", encoding="utf-8") as f:
                        backbone_cfg = json.load(f)
                else:
                    assert isinstance(
                        backbone_cfg, dict
                    ), "self-defined backbone config must be a dictionary."
        # from scratch
        else:
            try:
                if isinstance(backbone_cfg, str) and backbone_cfg.endswith(".json"):
                    with open(backbone_cfg, "r", encoding="utf-8") as f:
                        backbone_cfg = json.load(f)
                else:
                    assert isinstance(
                        backbone_cfg, dict
                    ), "self-defined backbone config must be a dictionary."
                self.hyena = HyenaDNAModel(
                    **backbone_cfg, 
                    use_head=use_head, 
                    n_classes=n_classes,
                )
            except TypeError as exc:
                raise TypeError("backbone_cfg must not be NoneType.") from exc

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
