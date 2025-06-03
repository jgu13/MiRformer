import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, AutoConfig, TrainingArguments, Trainer, DataCollatorWithPadding

###____________________
## Dataset
###____________________
# load dataset
dataset = load_dataset("csv", 
                       data_files="/home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/positive_samples_30_random_128_samples.csv",
                       delimiter='\t')
dataset['train'] = dataset['train'].remove_columns('label')
train_testvalid = dataset['train'].train_test_split(test_size=0.2)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})
# use DNABERT tokenizer
model_name = "zhihan1996/DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def preprocess(example):
    # Encode with truncation on the UTR only (context)
    encoding = tokenizer(
        example["miRNA sequence"],
        example["mRNA sequence"],
        truncation="only_second",
        padding="max_length",
        max_length=500,
        return_offsets_mapping=True
    )
    # Align character indices to token indices
    offsets = encoding["offset_mapping"]
    context_start = encoding["token_type_ids"].index(1)
    seed_start, seed_end = example["seed start"], example["seed end"]
    
    # Map char positions to token positions (for UTR portion only)
    start_idx, end_idx = 0, 0
    for idx, (start, end) in enumerate(offsets):
        if start <= seed_start < end:
            start_idx = idx
        if start < seed_end <= end:
            end_idx = idx

    #encoding["start_positions"] = start_idx
    #encoding["end_positions"] = end_idx

    encoding["labels"] = [start_idx, end_idx]
    encoding.pop("offset_mapping")  # Not needed for model
    return encoding

tokenized_dataset = dataset.map(preprocess)

###_______________________
## DNABERT Model w QA head
###_______________________
class DNABERTForQA(nn.Module):
    def __init__(self, 
                 model_name="zhihan1996/DNABERT-2-117M",
                 freeze_encoder=False):
        super(DNABERTForQA, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.dnabert = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=True)
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)  # outputs: (start_logits, end_logits)

    def forward(self, 
                input_ids, 
                attention_mask=None, 
                token_type_ids=None, 
                labels=None,
                ):
        outputs = self.dnabert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # print(len(outputs), outputs[0].shape, outputs[1].shape)
        # shape: (batch_size, seq_len, hidden_size)
        # sequence_output = outputs.last_hidden_state
        sequence_output = outputs[0]

        # shape: (batch_size, seq_len, 2)
        logits = self.qa_outputs(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)  # shape: (batch_size, seq_len, 1)
        start_logits = start_logits.squeeze(-1)  # shape: (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if labels is not None:
            # extract start position and end position
            start_positions = labels[:, 0]
            end_positions = labels[:, 1]
            loss_fn = nn.CrossEntropyLoss()
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            return {"loss": loss, "logits": (start_logits, end_logits)}
        else:
            return {"logits": (start_logits, end_logits)}

def compute_metrics(eval_pred):
    print("Compute metrics called")

    preds = eval_pred.predictions
    print("Predictions type:", type(preds))
    
    start_logits, end_logits = preds

    print("Start logits shape:", np.shape(start_logits))
    print("End logits shapr:", np.shape(end_logits))

    # check labels
    labels = eval_pred.label_ids
    
    start_labels, end_labels = labels[:,0], labels[:,1]

    print("Start labels shape:", np.shape(start_labels))
    print("End labels shape:", np.shape(end_labels))

    #start_logits, end_logits = eval_pred.predictions
    #start_labels = eval_pred.label_ids['start_positions']
    #end_labels = eval_pred.label_ids['end_positions']

    # Get predicted start/end positions
    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)

    # Compute exact match
    exact_matches = (start_preds == start_labels) & (end_preds == end_labels)
    accuracy = np.mean(exact_matches)
    print("Accuracy = ", accuracy)

    return {"accuracy": accuracy}

model = DNABERTForQA(freeze_encoder=False)

args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=32,
    num_train_epochs=100,
    learning_rate=5e-5,
    evaluation_strategy="epoch",
    logging_dir="./dna_bert_logs",
    save_strategy="epoch",
    fp16=True,
    logging_strategy="steps",         # log every N steps
    logging_steps=10,                 # how often to log 
    report_to="wandb",                # make sure logs go to W&B
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

print("Using device:", torch.cuda.current_device())

trainer.train()
