"""
Just a simple character level tokenizer.

From: https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py

CharacterTokenzier for Hugging Face Transformers.
This is heavily inspired from CanineTokenizer in transformers package.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import Optional
from functools import partial
from torch import Tensor
from torchvision.ops import StochasticDepth
from collections import namedtuple
from sklearn.model_selection import train_test_split

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        characters: Sequence[str],
        model_max_length: int,
        padding_side: str = "left",
        **kwargs
    ):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)


"""
The GenomicBenchmarks dataset will automatically download to /contents on colab.
There are 8 datasets to choose from.

"""

from random import random
import numpy as np
from pathlib import Path


# helper functions
def exists(val):
    return val is not None


def coin_flip():
    return random() > 0.5


string_complement_map = {
    "A": "U",
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "a": "t",
    "a": "u",
    "c": "g",
    "g": "c",
    "t": "a",
}


# augmentation
def string_reverse_complement(seq):
    rev_comp = ""
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


class GenomicBenchmarkDataset(torch.utils.data.Dataset):
    """
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.

    Genomic Benchmarks Dataset, from:
    https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks


    """

    def __init__(
        self,
        split,
        max_length,
        dataset_name="human_enhancers_cohn",
        d_output=2,  # default binary classification
        dest_path="/content",  # default for colab
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug

        # use Path object
        base_path = Path(dest_path) / dataset_name / split

        self.all_paths = []
        self.all_labels = []
        label_mapper = {}

        for i, x in enumerate(base_path.iterdir()):
            label_mapper[x.stem] = i

        for label_type in label_mapper.keys():
            for x in (base_path / label_type).iterdir():
                self.all_paths.append(x)
                self.all_labels.append(label_mapper[label_type])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        txt_path = self.all_paths[idx]
        with open(txt_path, "r") as f:
            content = f.read()
        x = content
        y = self.all_labels[idx]

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(
            x,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )  # add cls and eos token (+2)
        seq = seq["input_ids"]  # get input_ids

        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            seq.append(self.tokenizer.sep_token_id)

        # convert to tensor
        seq = torch.LongTensor(seq)

        # need to wrap in list
        target = torch.LongTensor([y])

        return seq, target

class miRawDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe,
        mRNA_max_length=40,
        miRNA_max_length=26,
        mRNA_col="mRNA sequence",
        miRNA_col="miRNA sequence",
        tokenizer=None,
        use_padding=None,
        rc_aug=None,
        add_eos=False,
        concat=False,
        add_linker=False,
    ):
        self.use_padding = use_padding
        self.mRNA_max_length = mRNA_max_length
        self.miRNA_max_length = miRNA_max_length
        self.tokenizer = tokenizer
        self.rc_aug = rc_aug
        self.add_eos = add_eos
        self.concat = concat
        self.data = dataframe
        self.add_linker = add_linker
        # Assuming the last column is the label, adjust this if needed
        self.mRNA_sequences = self.data[[mRNA_col]].values
        self.miRNA_sequences = self.data[[miRNA_col]].values
        self.labels = self.data[["label"]].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert data to PyTorch tensors
        mRNA_seq = self.mRNA_sequences[idx][0]
        miRNA_seq = self.miRNA_sequences[idx][0]
        labels = self.labels[idx]

        # replace U with T
        miRNA_seq = miRNA_seq.replace("U", "T")
        # reverse miRNA to 3' to 5'
        miRNA_seq = miRNA_seq[::-1]

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            mRNA_seq = string_reverse_complement(mRNA_seq)
            miRNA_seq = string_reverse_complement(miRNA_seq)

        mRNA_seq_encoding = self.tokenizer(
            mRNA_seq,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.mRNA_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        mRNA_seq_tokens = mRNA_seq_encoding["input_ids"]  # get input_ids
        mRNA_seq_mask = mRNA_seq_encoding["attention_mask"]  # get attention mask
        # print("Tokenized sequence length = ", len(mRNA_seq_tokens))

        miRNA_seq_encoding = self.tokenizer(
            miRNA_seq,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.miRNA_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        miRNA_seq_tokens = miRNA_seq_encoding["input_ids"]  # get input_ids
        miRNA_seq_mask = miRNA_seq_encoding["attention_mask"]  # get attention mask

        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            mRNA_seq_tokens.append(self.tokenizer.sep_token_id)
            miRNA_seq_tokens.append(self.tokenizer.sep_token_id)
            mRNA_seq_mask.append(1)  # do not mask eos token
            miRNA_seq_mask.append(1)  # do not mask eos token
        
        if self.concat:
            if self.add_linker:
                linker = [self.tokenizer._convert_token_to_id("N")] * 6
                concat_seq_tokens = mRNA_seq_tokens + linker + miRNA_seq_tokens
                linker_mask = [1] * 6
                concat_seq_mask = mRNA_seq_mask + linker_mask + miRNA_seq_mask
            else:
                # concatenate miRNA and mRNA tokens
                concat_seq_tokens = mRNA_seq_tokens + miRNA_seq_tokens
                concat_seq_mask = mRNA_seq_mask + miRNA_seq_mask
            # convert to tensor
            concat_seq_tokens = torch.tensor(concat_seq_tokens, dtype=torch.long)
            concat_seq_mask = torch.tensor(concat_seq_mask, dtype=torch.long)
            target = torch.tensor([labels], dtype=torch.float)

            return concat_seq_tokens, concat_seq_mask, target
        else:
            # convert to tensor
            mRNA_seq_tokens = torch.tensor(mRNA_seq_tokens, dtype=torch.long)
            miRNA_seq_tokens = torch.tensor(miRNA_seq_tokens, dtype=torch.long)
            mRNA_seq_mask = torch.tensor(mRNA_seq_mask, dtype=torch.long)
            miRNA_seq_mask = torch.tensor(miRNA_seq_mask, dtype=torch.long)
            target = torch.tensor([labels], dtype=torch.float)
            return mRNA_seq_tokens, miRNA_seq_tokens, mRNA_seq_mask, miRNA_seq_mask, target
