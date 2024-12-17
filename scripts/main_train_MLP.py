import torch.optim as optim
import torch
import torch.nn.functional as F
import json
import os
import pandas as pd
import subprocess
import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from Hyena_layer import *
from Data_pipeline import *
import time

"""
We provide simple training code for the GenomicBenchmark datasets.
"""

PROJ_HOME = os.path.expanduser('~/projects/mirLM')

def compute_cross_attention(Q, K, V, Q_mask, K_mask):
    d_model = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model) # [batchsize, mRNA_seq_len, miRNA_seq_len]
    # expand K mask to mask out keys, make each query only attend to valid keys
    K_mask = K_mask.unsqueeze(1).expand(-1, Q.shape[1], -1) # [batchsize, mRNA_seq_len, miRNA_seq_len]
    scores = scores.masked_fill(K_mask==0, -1e9)
    # apply softmax on the key dimension
    attn_weights = F.softmax(scores, dim=-1) # [batchsize, mRNA_seq_len, miRNA_seq_len]
    cross_attn = torch.matmul(attn_weights, V) # [batchsize, mRNA_seq_len, d_model]
    # expand Q mask to mask out queries, zero out padded queries
    valid_counts = Q_mask.sum(dim=1, keepdim=True) # [batchsize, 1] 
    Q_mask = Q_mask.unsqueeze(-1).expand(-1, -1, d_model) # [batchsize, mRNA_seq_len, d_model]
    cross_attn = cross_attn * Q_mask
    # average pool over seq_length
    cross_attn = cross_attn.sum(dim=1) / valid_counts # [batchsize, d_model]
    # print("Cross attention shape = ", cross_attn.shape)
    return cross_attn

def train(HyenaDNA_feature_extractor, MLP_head, device, train_loader, optimizer, epoch, loss_fn, log_interval=10, epoch_loss=0.0, accumulation_step=None):
    """Training loop."""
    MLP_head.train()
    HyenaDNA_feature_extractor.eval()
    for batch_idx, (mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target) in enumerate(train_loader):
        mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target = mRNA_seq.to(device), miRNA_seq.to(device), mRNA_seq_mask.to(device), miRNA_seq_mask.to(device), target.to(device)
        mRNA_hidden_states = HyenaDNA_feature_extractor(device=device, input_ids=mRNA_seq) # (batchsize, mRNA_seq_len, hidden), mask size = (batchsize, mRNA_seq_len)
        miRNA_hidden_states = HyenaDNA_feature_extractor(device=device, input_ids=miRNA_seq) # (batchsize, miRNA_seq_len, hidden), mask size = (batchsize, miRNA_seq_len)
        # cross attn between mRNA and miRNA embeddings
        Q = mRNA_hidden_states # [batchsize, mRNA_seq_len, d_model]
        K = miRNA_hidden_states # [batchsize, miRNA_seq_len, d_model]
        V = miRNA_hidden_states # [batchsize, miRNA_seq_len, d_model]
        Q_mask = mRNA_seq_mask
        K_mask = miRNA_seq_mask
        cross_attn = compute_cross_attention(Q=Q, K=K, V=V, Q_mask=Q_mask, K_mask=K_mask)
        output = MLP_head(cross_attn) # (batch_size, 1)
        loss = loss_fn(output.squeeze().sigmoid(), target.squeeze())
        if accumulation_step is not None:
            loss = loss / accumulation_step
            loss.backward()
            if batch_idx % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                        epoch, batch_idx * len(mRNA_seq), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                        epoch, batch_idx * len(mRNA_seq), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        epoch_loss += loss.item()
    average_loss = epoch_loss / len(train_loader)
    return average_loss
    
def test(HyenaDNA_feature_extractor, MLP_head, device, test_loader, loss_fn):
    """Test loop."""
    MLP_head.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target) in test_loader:
            mRNA_seq, miRNA_seq, mRNA_seq_mask, miRNA_seq_mask, target = mRNA_seq.to(device), miRNA_seq.to(device), mRNA_seq_mask.to(device), miRNA_seq_mask.to(device), target.to(device)
            mRNA_hidden_states = HyenaDNA_feature_extractor(device=device, input_ids=mRNA_seq)
            miRNA_hidden_states = HyenaDNA_feature_extractor(device=device, input_ids=miRNA_seq)
            Q = mRNA_hidden_states # [batchsize, mRNA_seq_len, d_model]
            K = miRNA_hidden_states # [batchsize, miRNA_seq_len, d_model]
            V = miRNA_hidden_states # [batchsize, miRNA_seq_len, d_model]
            Q_mask = mRNA_seq_mask
            K_mask = miRNA_seq_mask
            cross_attn = compute_cross_attention(Q=Q, K=K, V=V, Q_mask=Q_mask, K_mask=K_mask)
            output = MLP_head(cross_attn) # (batch_size, 1)
            probabilities = torch.sigmoid(output.squeeze())
            predictions = (probabilities > 0.5).long()
            correct += predictions.eq(target.squeeze()).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        accuracy))
    return accuracy

def load_external_dataset(external_dataset):
    '''
    `external_dataset` can be a path to the locally saved dataset or it can be an loaded pandas DataFrame
    '''
    # if external_dataset is a path to a file
    if isinstance(external_dataset, str):
        if external_dataset.endswith('.csv'):
            D = pd.read_csv(external_dataset, sep = ',')
        elif external_dataset.endswith('.tsv'):
            D = pd.read_csv(external_dataset, sep='\t')
        elif external_dataset.endswith('.txt'):
            D = pd.read_csv(external_dataset, sep='\t')
        elif external_dataset.endswith('.xlsx'):
            D = pd.read_excel(external_dataset)
        elif external_dataset.endswith('.json'):
            D = pd.read_csv(external_dataset)
    # if external_dataset is a pandas dataframe
    elif isinstance(external_dataset, pd.DataFrame):
        D = external_dataset
    return D

def run_train(mRNA_length, miRNA_max_length, epochs, external_dataset=None):

    '''
    Main entry point for training. 
    '''
    # experiment settings:
    num_epochs = epochs  # ~100 seems fine
    mRNA_max_length = mRNA_length  # max len of sequence of dataset (of what you want)
    miRNA_max_length = miRNA_max_length
    use_padding = True
    batch_size = 16
    accumulation_step = 256 / batch_size
    rc_aug = False  # reverse complement augmentation
    add_eos = True  # add end of sentence token

    # for fine-tuning, only the 'tiny' model can fit on colab
    pretrained_model_name = 'hyenadna-small-32k-seqlen-hf'  # use None if training from scratch

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 1

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one by default
    backbone_cfg = None
    download = False

    device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen', 'hyenadna-small-32k-seqlen-hf', 'hyenadna-medium-160k-seqlen-hf', 'hyenadna-medium-450k-seqlen-hf', 'hyenadna-large-1m-seqlen-hf']:
        path = f'{PROJ_HOME}/checkpoints'
        # use the pretrained Huggingface wrapper instead
        HyenaDNA_feature_extractor = HyenaDNAPreTrainedModel.from_pretrained(
            path,
            pretrained_model_name,
            download=download,
            config=backbone_cfg,
            device=device,
            use_head=use_head
        )
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, pretrained_model_name)
        if os.path.isdir(pretrained_model_name_or_path):
            if backbone_cfg is None:
                backbone_cfg = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
    # from scratch
    elif backbone_cfg:
        HyenaDNA_feature_extractor = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)
    
    learning_rate = backbone_cfg['lr']
    weight_decay = backbone_cfg['weight_decay']
    hidden_sizes = [backbone_cfg['d_model']*2, backbone_cfg['d_model']*2]
    MLP_head = LinearHead(d_model=backbone_cfg['d_model'], d_output=n_classes, hidden_sizes=hidden_sizes)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'U', 'N'],  # add RNA characters, N is uncertain
        model_max_length=mRNA_max_length + miRNA_max_length + 2,  # to account for miRNA length and special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left' # since HyenaDNA is causal, we pad on the left
    )

    D = load_external_dataset(external_dataset)

    ds_train, ds_test = train_test_split(D, test_size=0.2, random_state=34, shuffle=True)
    ds_train = CustomDataset(ds_train,
                            mRNA_max_length=mRNA_max_length,
                            miRNA_max_length=miRNA_max_length,
                            tokenizer=tokenizer,
                            use_padding=use_padding,
                            rc_aug=rc_aug,
                            add_eos=add_eos)
    ds_test = CustomDataset(ds_test,
                            mRNA_max_length=mRNA_max_length,
                            miRNA_max_length=miRNA_max_length,
                            tokenizer=tokenizer,
                            use_padding=use_padding,
                            rc_aug=rc_aug,
                            add_eos=add_eos)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # loss function
    loss_fn = nn.BCELoss()

    # create optimizer
    # only optimize MLP layers
    optimizer = optim.AdamW(MLP_head.parameters(), lr=learning_rate, weight_decay=weight_decay)

    HyenaDNA_feature_extractor.to(device)
    MLP_head.to(device)
    print(MLP_head)
    print("Leanring rate = ", learning_rate)
    print("weight decay = ", weight_decay)

    average_loss_list = []
    accuracy_list = []
    
    model_checkpoints_dir = os.path.join(PROJ_HOME, 'checkpoints', 'mirLM', f'mirLM-{mRNA_length}')
    if not os.path.isdir(model_checkpoints_dir):
        os.mkdir(model_checkpoints_dir)
    
    for epoch in range(num_epochs):
        average_loss = train(HyenaDNA_feature_extractor=HyenaDNA_feature_extractor, 
                             MLP_head=MLP_head, 
                             device=device, 
                             train_loader=train_loader, 
                             optimizer=optimizer, 
                             epoch=epoch, 
                             loss_fn=loss_fn, 
                             accumulation_step=accumulation_step)
        accuracy = test(HyenaDNA_feature_extractor=HyenaDNA_feature_extractor, 
                                   MLP_head=MLP_head, 
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
                'model_state_dict': MLP_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
                'accuracy': accuracy
            }, os.path.join(model_checkpoints_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # save the final model
    torch.save({
                'epoch': epoch,
                'model_state_dict': MLP_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
                'accuracy': accuracy,
            }, os.path.join(model_checkpoints_dir, f'checkpoint_epoch_final.pth'))
    
    return average_loss_list, accuracy_list


if __name__ == "__main__":
    mRNA_length = 1000
    miRNA_length = 28
    epochs = 1
    dataset = "mirLM"
    model = "MLP"
    print("Binary Classification -- Start training --")
    print("mRNA length: ", mRNA_length)
    print(f"For {epochs} epochs")
    # load dummmy test dataset
    external_data_path = os.path.join(PROJ_HOME, 'data', f"training_{mRNA_length}_random_256_samples.csv")

    # launch it
    start = time.time()
    train_average_loss, test_accuracy = run_train(mRNA_length=mRNA_length, miRNA_max_length=miRNA_length, epochs=epochs, external_dataset=external_data_path)
    time_taken = time.time() - start
    print("Time taken for {} epoch = {} min.".format(epochs, time_taken/60))
    
    # # save test_accuracy
    # perf_dir = os.path.join(PROJ_HOME, "Performance", dataset, model)
    # with open(os.path.join(perf_dir, f"test_accuracy_{mRNA_length}.json"), "w") as fp:
    #     json.dump(test_accuracy, fp)
    # with open(os.path.join(perf_dir, f"train_loss_{mRNA_length}.json"), "w") as fp:
    #     json.dump(train_average_loss, fp)