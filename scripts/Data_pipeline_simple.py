"""
Simplified CharacterTokenizer that avoids transformers inheritance issues.
"""

import os
from typing import Dict, List, Optional, Sequence, Union, Tuple

class SimpleCharacterTokenizer:
    """A simple character tokenizer without complex transformers inheritance."""
    
    def __init__(
        self,
        characters: Sequence[str],
        model_max_length: int,
        padding_side: str = "left",
        **kwargs
    ):
        """Character tokenizer for simple use cases.
        Args:
            characters (Sequence[str]): List of desired characters
            model_max_length (int): Model maximum sequence length
            padding_side (str): Which side to pad sequences
        """
        self.characters = characters
        self.model_max_length = model_max_length
        self.padding_side = padding_side
        
        # Define special tokens
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.bos_token = "[BOS]"
        self.mask_token = "[MASK]"
        self.pad_token = "[PAD]"
        self.reserved_token = "[RESERVED]"
        self.unk_token = "[UNK]"
        self.mrna_cls_token = "[MRNA_CLS]"
        
        # Create vocabulary
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            "[MRNA_CLS]": 7,
            **{ch: i + 8 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        
        # Set token IDs as properties
        self.cls_token_id = self._vocab_str_to_int[self.cls_token]
        self.sep_token_id = self._vocab_str_to_int[self.sep_token]
        self.bos_token_id = self._vocab_str_to_int[self.bos_token]
        self.mask_token_id = self._vocab_str_to_int[self.mask_token]
        self.pad_token_id = self._vocab_str_to_int[self.pad_token]
        self.unk_token_id = self._vocab_str_to_int[self.unk_token]
        self.mrna_cls_token_id = self._vocab_str_to_int[self.mrna_cls_token]

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into characters."""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        return self._vocab_int_to_str[index]

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary mapping token strings to token ids."""
        return self._vocab_str_to_int.copy()

    def convert_tokens_to_string(self, tokens):
        """Convert tokens back to string."""
        return "".join(tokens)

    def encode_plus(self, text: str, **kwargs):
        """Encode text and return a dictionary with token ids and attention mask."""
        tokens = self._tokenize(text)
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "token_type_ids": [0] * len(token_ids)
        }

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token ids back to text."""
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        return self.convert_tokens_to_string(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build inputs with special tokens."""
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
        """Get special tokens mask."""
        if already_has_special_tokens:
            return [1 if token_id in [self.cls_token_id, self.sep_token_id] else 0 
                   for token_id in token_ids_0]

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create token type IDs from sequences."""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the tokenizer to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.txt")
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token, token_id in sorted(self._vocab_str_to_int.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{token_id}\n")
        
        # Save config
        config = {
            "characters": self.characters,
            "model_max_length": self.model_max_length,
            "padding_side": self.padding_side,
            "vocab_size": self.vocab_size
        }
        
        import json
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        return save_directory

    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs):
        """Load the tokenizer from a directory."""
        # Load config
        import json
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Create tokenizer
        tokenizer = cls(
            characters=config["characters"],
            model_max_length=config["model_max_length"],
            padding_side=config["padding_side"]
        )
        
        return tokenizer
