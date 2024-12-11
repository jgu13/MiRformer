import torch.optim as optim
import torch
import json
import os
import pandas as pd
import subprocess
import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from sklearn.model_selection import train_test_split
from Hyena_layer import *
from Data_pipeline import *
import time

"""
We provide simple training code for the GenomicBenchmark datasets.
"""

PROJ_HOME = os.path.expanduser('~/projects/mirLM')


def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=10, epoch_loss=0.0, accumulation_step=None):
    """Training loop."""
    model.train()
    for batch_idx, (seq, seq_mask, target) in enumerate(train_loader):
        seq, seq_mask, target = seq.to(device), seq_mask.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(device=device, input_ids=seq, input_mask=seq_mask)
        loss = loss_fn(output.squeeze().sigmoid(), target.squeeze())
        if accumulation_step is not None:
            loss = loss / accumulation_step
            loss.backward()
            if batch_idx % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                        epoch, batch_idx * len(seq), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                        epoch, batch_idx * len(seq), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        epoch_loss += loss.item()
    average_loss = epoch_loss / len(train_loader)
    return average_loss
    
    
def test(model, device, test_loader, loss_fn):
    """Test loop."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for seq, seq_mask, target in test_loader:
            seq, target = seq.to(device), target.to(device)
            output = model(device=device, input_ids=seq, input_mask=seq_mask)
            probabilities = torch.sigmoid(output.squeeze())
            predictions = (probabilities > 0.5).long()
            correct += predictions.eq(target.squeeze()).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy

def load_data(dataset=None, sep=','):
    if dataset:
        # if dataset is a path to a file
        if isinstance(dataset, str):
            if dataset.endswith('.csv') or dataset.endswith('.txt') or dataset.endswith('.tsv'):
                D = pd.read_csv(dataset, sep=sep)
            elif dataset.endswith('.xlsx'):
                D = pd.read_excel(dataset, sep=sep)
            elif dataset.endswith('.json'):
                with open(dataset, 'r') as f:
                    D = json.load(f)
            else:
                print(f"Unrecognized format of {dataset}")
                D = None
        # if dataset is a pandas dataframe
        elif isinstance(dataset, pd.DataFrame):
            D = dataset
        else:
            print("Dataset must be a path or a pandas dataframe.")
            D = None
    else:
        print(f"Dataset is {dataset}.")
        D = None
    return D

def run_train(mRNA_max_len, miRNA_max_len, dataset=None, epochs=10):

    '''
    Main entry point for training.  Select the dataset name and metadata, as
    well as model and training args, and you're off to the genomic races!

    ### GenomicBenchmarks Metadata
    # there are 8 datasets in this suite, choose 1 at a time, with their corresponding settings
    # name                                num_seqs        num_classes     median len    std
    # dummy_mouse_enhancers_ensembl       1210            2               2381          984.4
    # demo_coding_vs_intergenomic_seqs    100_000         2               200           0
    # demo_human_or_worm                  100_000         2               200           0
    # human_enhancers_cohn                27791           2               500           0
    # human_enhancers_ensembl             154842          2               269           122.6
    # human_ensembl_regulatory            289061          3               401           184.3
    # human_nontata_promoters             36131           2               251           0
    # human_ocr_ensembl                   174756          2               315           108.1

    '''
    # experiment settings:
    num_epochs = epochs  # ~100 seems fine
    max_length = mRNA_max_len + miRNA_max_len + 2  # max len of sequence of dataset (of what you want)
    use_padding = True
    batch_size = 16
    learning_rate = 3e-4  # good default for Hyena
    rc_aug = False  # reverse complement augmentation
    add_eos = False  # add end of sentence token
    weight_decay = 0.1
    accumulation_step = 256 / batch_size # effectively change batch size to 256

    # for fine-tuning, only the 'tiny' model can fit on colab
    pretrained_model_name = 'hyenadna-small-32k-seqlen-hf'  # use None if training from scratch

    # we need these for the decoder head, if using
    use_head = True
    n_classes = 1

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one by default
    backbone_cfg = None
    
    mRNA_max_length = 500
    miRNA_max_length = 28

    device = 'cuda:9' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen', 'hyenadna-small-32k-seqlen-hf', 'hyenadna-medium-160k-seqlen-hf', 'hyenadna-medium-450k-seqlen-hf', 'hyenadna-large-1m-seqlen-hf']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            f'{PROJ_HOME}/checkpoints',
            pretrained_model_name,
            download=False,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )

    # from scratch
    else:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'U', 'N'],  # add RNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )

    D = load_data(dataset, sep=",")
    
    ds_train, ds_test = train_test_split(D, test_size=0.2, random_state=34, shuffle=True)
    ds_train = miRawDataset(ds_train,
                            mRNA_max_length=mRNA_max_length,
                            miRNA_max_length=miRNA_max_length,
                            tokenizer=tokenizer,
                            use_padding=use_padding,
                            rc_aug=rc_aug,
                            add_eos=add_eos)
    ds_test = miRawDataset(ds_test,
                            mRNA_max_length=mRNA_max_length,
                            miRNA_max_length=miRNA_max_length,
                            tokenizer=tokenizer,
                            use_padding=use_padding,
                            rc_aug=rc_aug,
                            add_eos=add_eos)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    # loss function
    loss_fn = nn.BCELoss()

    # create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(device)

    average_loss_list = []
    accuracy_list = []
    
    for epoch in range(num_epochs):
        average_loss = train(model=model, 
                             device=device, 
                             train_loader=train_loader, 
                             optimizer=optimizer, 
                             epoch=epoch, 
                             loss_fn=loss_fn, 
                             accumulation_step=accumulation_step)
        accuracy = test(model=model, 
                        device=device, 
                        test_loader=test_loader, 
                        loss_fn=loss_fn)
        optimizer.step()
        average_loss_list.append(average_loss)
        accuracy_list.append(accuracy)
        
        # save checkpoints
        if epoch % 10 == 0:
            # Save the model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
                'accuracy': accuracy,
            }, os.path.join(PROJ_HOME, 'checkpoints', 'mirLM', f'mirLM-500-finetune-Hyena', f'checkpoint_epoch_{epoch}.pth'))
    # save the final model
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
                'accuracy': accuracy,
            }, os.path.join(PROJ_HOME, 'checkpoints', 'mirLM', f'mirLM-500-finetune-Hyena', f'checkpoint_epoch_final.pth'))
    
    return average_loss_list, accuracy_list
        
if __name__ == "__main__":
    mRNA_length = 5000
    miRNA_length = 28
    epochs = 10
    dataset = "mirLM"
    model = "HyenaDNA"
    print("Binary Classification -- Start training --")
    print("mRNA length: ", mRNA_length)
    print(f"For {epochs} epochs")
    # load dummmy test dataset
    data_path = os.path.join(PROJ_HOME, 'data', f'training_{mRNA_length}.csv')

    # launch it
    # start = time.time()
    train_average_loss, test_accuracy = run_train(mRNA_max_len=mRNA_length, miRNA_max_len=miRNA_length, dataset=data_path, epochs=epochs)
    # time_taken = time.time() - start
    # print("Time taken for one epoch is {:.2f} min".format(time_taken/60))
        
    # save test_accuracy
    perf_dir = os.path.join(PROJ_HOME, "Performance", dataset, model)
    with open(os.path.join(perf_dir, f"test_accuracy_{mRNA_length}.json"), "w") as fp:
        json.dump(test_accuracy, fp)
    with open(os.path.join(perf_dir, f"train_loss_{mRNA_length}.json"), "w") as fp:
        json.dump(train_average_loss, fp)