import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import Transformer, TransformerConfig

# Custom Dataset Class for Sequence Data
class SequenceDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        """
        Args:
            input_sequences: List of input sequences (e.g., tokenized sentences)
            target_sequences: Corresponding target sequences shifted by one position
        """
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.input_sequences[idx]), torch.tensor(self.target_sequences[idx])


# Transformer Model Wrapper
class TransformerForAutoregressiveGeneration(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers, max_seq_len):
        """
        Args:
            vocab_size: Size of the vocabulary
            model_dim: Dimensionality of the embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=0.1,
        )
        self.output_layer = nn.Linear(model_dim, vocab_size)

    def create_positional_encoding(self, max_seq_len, model_dim):
        """Creates sinusoidal positional encodings."""
        pe = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids, target_ids):
        """
        Forward pass for the model.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            target_ids: Target token IDs (batch_size, seq_len)

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
        """
        # Add embeddings and positional encoding
        input_emb = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        target_emb = self.embedding(target_ids) + self.positional_encoding[:, :target_ids.size(1), :]

        # Generate causal mask to prevent future token leakage
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_ids.size(1)).to(input_ids.device)

        # Pass through the transformer model
        transformer_output = self.transformer(
            src=input_emb.permute(1, 0, 2),  # Transformer expects (seq_len, batch_size, model_dim)
            tgt=target_emb.permute(1, 0, 2),
            tgt_mask=tgt_mask,
        )

        # Project to vocabulary size
        logits = self.output_layer(transformer_output.permute(1, 0, 2))  # Back to (batch_size, seq_len, vocab_size)
        return logits


# Training Configuration
def train_model():
    # Hyperparameters
    vocab_size = 10000  # Example vocabulary size
    model_dim = 512
    num_heads = 8
    num_layers = 6
    max_seq_len = 50
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10

    # Sample input and target data
    input_sequences = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]  # Replace with actual tokenized data
    target_sequences = [[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]

    # Prepare DataLoader
    dataset = SequenceDataset(input_sequences, target_sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    model = TransformerForAutoregressiveGeneration(vocab_size, model_dim, num_heads, num_layers, max_seq_len)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.to(model.device), target_ids.to(model.device)

            # Shift target for teacher forcing
            target_input = target_ids[:, :-1]
            target_output = target_ids[:, 1:]

            # Forward pass
            logits = model(input_ids, target_input)
            loss = criterion(logits.view(-1, vocab_size), target_output.view(-1))  # Flatten for loss computation

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    print("Training complete!")


if __name__ == "__main__":
    train_model()
