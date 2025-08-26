# MIL Binding Implementation with Global Token for Longformer + Cross-Attention

This document describes the implementation of Multiple Instance Learning (MIL) with a global token for the Longformer + cross-attention pipeline in the mirLM project, focusing on binding prediction.

## Overview

The implementation adds a global token (CLS) to mRNA sequences and implements MIL binding pooling to improve binding prediction. The global token allows the model to attend to all positions in the mRNA sequence, while MIL pooling aggregates information across positions to make a final binding prediction.

## MIL Replacement of Binding Prediction

**Important**: With MIL, the old mean-pooled binding prediction has been completely replaced:

- **Before**: `binding_logits` (mean-pooled over mRNA positions) → `binding_probs = σ(binding_logits)`
- **After**: `bag_logit` (MIL aggregated over mRNA positions) → `p_bag = σ(bag_logit)`

**What this means**:
- `binding_logit` replaces `binding_logits` for all binding predictions
- `binding_probs = σ(binding_logit)` is calculated during evaluation (not in BindingHead)
- The old `binding_output` layer is no longer used
- All metrics, thresholds, and evaluation now use MIL predictions
- `binding_probs` replaces `binding_probs` everywhere

## Key Components

### 1. Global Token (MRNA_CLS)

- **Token ID**: Dynamically assigned by tokenizer (position 7 in vocabulary)
- **Position**: Always at index 0 in mRNA sequences
- **Purpose**: Enables global attention across the entire mRNA sequence
- **Implementation**: Added to tokenizer vocabulary as `[MRNA_CLS]` token

### 2. MIL Binding Head

The `BindingHead` class implements:
- **Seed Scorer**: Linear layer that scores each mRNA position
- **LSE Pooling**: Log-Sum-Exp pooling for smooth aggregation
- **Temperature Scaling**: Configurable temperature parameter (default: 0.1)
- **Attention Weights**: Returns position-wise attention weights for visualization

## Implementation Details

### Tokenizer Integration

The `CharacterTokenizer` has been extended to include the `[MRNA_CLS]` token:

```python
# Tokenizer automatically includes [MRNA_CLS] at position 7
tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"], ...)
print(f"Vocabulary size: {tokenizer.vocab_size}")  # 13 (7 special + 1 MRNA_CLS + 5 bases)
print(f"MRNA_CLS token ID: {tokenizer.mrna_cls_token_id}")  # 7
```

**Note**: The model automatically uses the correct vocabulary size when a tokenizer is provided. If no tokenizer is provided, it defaults to 13 tokens (which includes the MRNA_CLS token).

### Data Preprocessing

The `QuestionAnswerDataset` automatically:
1. Prepends the global token (MRNA_CLS_ID = 12) to mRNA sequences
2. Shifts seed start/end positions by +1 to account for the prepended token
3. Maintains proper attention masks

### Longformer Attention Mask

- **Convention**: -1=pad, 0=local, 1=global
- **Global Token**: Position 0 is set to 1 (global attention)
- **Verification**: Assertion ensures global token is properly set

### Model Architecture

```
Input: (B, Lm, D) mRNA embeddings with global token at position 0
       ↓
Longformer Self-Attention (with global attention on position 0)
       ↓
Cross-Attention (mRNA as Q, miRNA as K/V)
       ↓
MIL Binding Head: scores positions → LSE pooling → binding prediction
       ↓
Combined Loss: MIL + Span + Binding (if enabled)
```

## Usage

### Basic Usage

```python
from transformer_model import QuestionAnsweringModel
from Data_pipeline import CharacterTokenizer

# Create tokenizer with MRNA_CLS token
tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                               add_special_tokens=False,
                               model_max_length=520,
                               padding_side="right")

# Create model (tokenizer will be automatically used during training)
model = QuestionAnsweringModel(
    mrna_max_len=520,
    mirna_max_len=24,
    embed_dim=1024,
    num_heads=8,
    num_layers=4,
    ff_dim=4096,
    use_longformer=True,
    predict_span=True,
    predict_binding=True
)
```

### Training

The model automatically handles:
- MIL binding loss calculation
- Combined loss optimization
- Proper gradient flow through all components

### Inference

```python
# Get MIL binding attention weights for visualization
pos_weights, pos_logits = model.predictor.get_binding_attention_weights(z_mrna, mrna_mask)

# pos_weights: (B, Lm) - attention weights over mRNA positions
# pos_logits: (B, Lm) - raw logits for each position
```

## Expected Shapes

### Inputs
- `mrna_ids`: (B, Lm) with global token at position 0
- `mrna_mask`: (B, Lm) with 1 for valid tokens (including CLS), 0 for pad
- `longformer_attn_mask`: (B, Lm) with {-1,0,1}, position 0 is 1 (global), -1 for pad

### Outputs
- `binding_logit`: (B,) - MIL binding prediction (replaces old binding_logits)
- `binding_probs`: (B,) - MIL binding probabilities (replaces old binding_probs)
- `binding_aux`: dict with `pos_weights` and `pos_logits`
- `start_logits, end_logits`: (B, Lm) or None

## Hyperparameters

### MIL Parameters
- **Temperature (tau)**: 0.1 (start with 0.2 → anneal to 0.05 for sharper spikes)
- **Global Tokens**: 1 (the CLS at position 0)
- **Window Size**: Keep existing 2w=40 as configured

### Loss Weights
- **MIL Bag Loss**: Always included
- **Span Loss**: Weighted by `alpha2` parameter
- **Binding Loss**: Weighted by `alpha1` parameter

## Monitoring and Debugging

### Logging
- MIL binding accuracy is logged to wandb
- All metrics are printed during evaluation

### Sanity Checks
- Global token verification: `assert (lf_mask[:,0]==1).all()`
- MIL binding weights normalization: weights sum to 1 for valid positions
- Shape consistency checks throughout the pipeline
- MIL replacement: `binding_logit` and `binding_probs` are used instead of old binding predictions

### Visualization
- `pos_weights`: Heatmap of binding attention over mRNA positions
- Expect peaked weights over seed in positives, flat in negatives
- Use for ISM analysis and interpretability

## Testing

Run the test script to verify implementation:

```bash
cd scripts
python test_mil_implementation.py
```

This will test:
- Model creation and initialization
- Forward pass with global token
- MIL binding output shapes and normalization

## Troubleshooting

### Common Issues

1. **Global token not at position 0**: Check dataset preprocessing
2. **Shape mismatches**: Verify input/output shapes match documentation
3. **MIL weights not normalized**: Check temperature parameter and mask handling
4. **Longformer mask errors**: Ensure global token is set to 2

### Debug Steps

1. Verify `MRNA_CLS` token is properly added to tokenizer vocabulary
2. Check that seed positions are shifted by +1
3. Confirm Longformer attention mask has position 0 set to 1 (global)
4. Validate MIL binding output shapes and normalization
5. Ensure `binding_logit` and `binding_probs` are used instead of old `binding_logits`

## Performance Considerations

- **Memory**: Global attention increases memory usage
- **Computation**: MIL binding pooling adds minimal overhead
- **Training**: Combined losses may require learning rate tuning
- **Inference**: MIL binding attention weights available for visualization

## Future Enhancements

- **Multiple Global Tokens**: Support for multiple global attention positions
- **Attention Regularization**: Add regularization on MIL binding attention weights
- **Hierarchical MIL**: Multi-level MIL for complex biological structures
- **Interpretability**: Enhanced visualization and analysis tools


