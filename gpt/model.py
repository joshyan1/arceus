import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from termcolor import colored
from tqdm import tqdm
import os
import time
print("Importing libraries...")

### hyper params
# model
ctx_len = 128
n_emb = 128
dropout = 0.1
head_size = 128
n_heads = 4 
n_layers = 3

# training
num_epochs = 10
batch_size = 128
num_batches = 0
lr = 1e-3

print("Hyperparameters set:")
print(f"Context length: {ctx_len}")
print(f"Embedding size: {n_emb}")
print(f"Dropout: {dropout}")
print(f"Head size: {head_size}")
print(f"Number of heads: {n_heads}")
print(f"Number of layers: {n_layers}")
print(f"Number of epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")

### Tokenization
print("Loading and tokenizing data...")
data_path = os.environ.get('TRAINING_DATA_PATH', os.path.join(os.path.dirname(__file__), 'data.txt'))
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
itos = {i: c for i, c in enumerate(vocab)}  # int to string
stoi = {c: i for i, c in enumerate(vocab)}  # string to int
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])
data = encode(text)
split = int(0.9 * len(data))
train_data = data[:split]
num_batches = len(train_data) // batch_size
val_data = data[split:]

print(f"Vocabulary size: {vocab_size}")
print(f"Total data length: {len(data)}")
print(f"Train data length: {len(train_data)}")
print(f"Validation data length: {len(val_data)}")

### Data Prep
print("Preparing data...")
ctx_len = 8
X_train = torch.tensor([train_data[i:i+ctx_len] for i in range(0, len(train_data) - ctx_len, ctx_len)])
y_train = torch.tensor([train_data[i+1:i+ctx_len+1] for i in range(0, len(train_data) - ctx_len, ctx_len)])
X_val = torch.tensor([val_data[i:i+ctx_len] for i in range(0, len(val_data) - ctx_len, ctx_len)])
y_val = torch.tensor([val_data[i+1:i+ctx_len+1] for i in range(0, len(val_data) - ctx_len, ctx_len)])

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

def get_batches(X, y, b_size, shuffle=True):
    if shuffle:
        ix = np.arange(X.shape[0])
        np.random.shuffle(ix)
        ix = torch.tensor(ix)
        X = X[ix]
        y = y[ix]
    for i in range(0, X.shape[0], b_size):
        input = X[i:i+b_size]
        label = y[i:i+b_size]
        yield input, label

print("Batch generation function defined.")

# Calculate number of batches per epoch
num_batches_per_epoch = math.ceil(X_train.shape[0] / batch_size)
print(f"Number of batches per epoch: {num_batches_per_epoch}")

### Model Definition
print("Defining model architecture...")

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_emb)
        self.wpe = nn.Embedding(ctx_len, n_emb)
        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layers)],
        )
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        self._init_parameters()
        print("GPT model initialized.")

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, max_new_tokens, device):
        print(f"Generating {max_new_tokens} new tokens...")
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits = self(ctx[:, -ctx_len:])
            logits = logits[:, -1, :]
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ctx = torch.cat((ctx, next_tok), dim=1)
        print("Generation complete.")
        return ctx

    def _init_parameters(self):
        print("Initializing model parameters...")
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            if 'bias' in name:
                nn.init.zeros_(param)
        print("Parameter initialization complete.")


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(n_emb, head_size, bias=False)
        self.q_proj = nn.Linear(n_emb, head_size, bias=False)
        self.v_proj = nn.Linear(n_emb, head_size, bias=False)
        self.causal_mask = torch.tril(torch.ones(ctx_len, ctx_len)).unsqueeze(0).unsqueeze(0) * -1e9
        self.c_proj = nn.Linear(head_size, n_emb)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        print("MultiHeadAttention module initialized.")

    def forward(self, x):
        B, T, C = x.shape
        
        K = self.k_proj(x)
        Q = self.q_proj(x)
        V = self.v_proj(x)

        K = K.view(B, T, n_heads, head_size // n_heads).transpose(1, 2)
        Q = Q.view(B, T, n_heads, head_size // n_heads).transpose(1, 2)
        V = V.view(B, T, n_heads, head_size // n_heads).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn_weights = attn_weights + (1 - causal_mask) * -1e9
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        o = torch.matmul(attn_weights, V)
        o = o.transpose(1, 2).contiguous().view(B, T, head_size)
        o = self.c_proj(self.resid_dropout(o))
        
        return o


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_emb, 4 * n_emb)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)
        print("MLP module initialized.")

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.mha = MultiHeadAttention()
        self.ln_1 = nn.LayerNorm(n_emb)
        self.ln_2 = nn.LayerNorm(n_emb)
        print("Block module initialized.")

    def forward(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


### Training
print("Defining loss function...")
def loss_fn(model, x, y):
    logits = model(x)
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    y = y.view(B * T)
    loss = F.cross_entropy(logits, y)
    return loss

print("Setting up device...")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

print("Initializing model and optimizer...")
model = GPT().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Calculate total parameters
total_params = sum(p.numel() for p in model.parameters())

def train(message_queue=None, job_id='training_room'):
    print("\nStarting training loop...")
    progress_bar = tqdm(range(num_epochs), desc="Training Progress")
    epoch_start = time.time()
    total_tokens_trained = 0

    for epoch in progress_bar:
        model.train()
        running_loss = 0
        batch_cnt = 0
        
        # Training loop with inner progress bar
        batch_progress = tqdm(get_batches(X_train, y_train, batch_size), 
                            desc=colored(f"Epoch {epoch+1}", "cyan"),
                            leave=False)
        
        for batch_idx, (input, label) in enumerate(batch_progress):
            batch_start = time.time()
            
            # Time the data preparation
            start_prep = time.time()
            input, label = input.to(device), label.to(device)
            prep_time = time.time() - start_prep
            
            # Time the forward pass
            start_forward = time.time()
            optimizer.zero_grad()
            logits = model(input)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            label = label.view(B * T)
            loss = F.cross_entropy(logits, label)
            forward_time = time.time() - start_forward
            
            # Time the backward pass
            start_backward = time.time()
            loss.backward()
            backward_time = time.time() - start_backward
            
            # Time the parameter update
            start_update = time.time()
            optimizer.step()
            update_time = time.time() - start_update
            
            running_loss += loss.item()
            batch_cnt += 1
            total_tokens_trained += B * T
            batch_time = time.time() - batch_start
            
            if batch_idx % 20 == 0 and message_queue is not None:
                # First send timing stats
                message_queue.put({
                    'event': 'timing_stats',
                    'data': {
                        'batch_idx': batch_idx,
                        'avg_forward': forward_time,
                        'avg_backward': backward_time,
                        'avg_update': update_time,
                        'avg_prep': prep_time,
                        'avg_comm': 0,  # No communication overhead for single-device training
                        'device_data': [{
                            'device_id': 1,
                            'total_teraflops': (total_params * 2 * batch_size) / (forward_time + backward_time) / 1e12,
                            'chip': 'CPU' if device.type == 'cpu' else 'GPU'
                        }],
                        'total_tokens_trained': total_tokens_trained
                    },
                    'room': job_id
                })

                # Then send training data
                message_queue.put({
                    'event': 'training_data',
                    'data': {
                        'epoch': epoch,
                        'epochs': num_epochs,
                        'train_loss': float(loss.item()),
                        'train_acc': 0,
                        'batch_idx': batch_idx,
                        'batch_time': batch_time,
                        'total_tokens_trained': total_tokens_trained,
                        'num_batches': num_batches,
                        'tokens_trained': B * T
                    },
                    'room': job_id
                })
        
        # Validation
        model.eval()
        val_loss = 0
        val_cnt = 0
        
        with torch.no_grad():
            for input, label in get_batches(X_val, y_val, batch_size):
                input, label = input.to(device), label.to(device)
                logits = model(input)
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                label = label.view(B * T)
                val_loss += F.cross_entropy(logits, label).item()
                val_cnt += 1
        
        val_loss /= val_cnt
        epoch_time = time.time() - epoch_start
        
        if message_queue is not None:
            message_queue.put({
                'event': 'epoch_stats',
                'data': {
                    'epoch': epoch,
                    'epochs': num_epochs,
                    'train_loss': float(running_loss / batch_cnt),
                    'val_loss': float(val_loss),
                    'epoch_time': epoch_time,
                    'total_tokens_trained': total_tokens_trained
                },
                'room': job_id
            })
        
        epoch_start = time.time()

if __name__ == "__main__":
    # When run directly, train without message queue
    train()
