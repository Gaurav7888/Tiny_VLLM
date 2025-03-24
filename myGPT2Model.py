import torch
import torch.nn as nn
import math
from transformers import GPT2Config
from transformers.activations import gelu_new

GPT2_CONFIG_PATH = './static_files/config.json'
GPT2_WEIGHTS_PATH = './static_files/gpt_pytorch_model.bin'

class FeedForwardNetwork(nn.Module):
    def __init__(self, config, model_weights, layer_idx):
        super().__init__()
        self.config = config
        self.model_weights = model_weights
        self.layer_idx = layer_idx
        self.fc = nn.Linear(self.config.n_embd, 4 * self.config.n_embd)
        self.act = gelu_new
        self.proj = nn.Linear(4 * self.config.n_embd, self.config.n_embd)
        self.load_weights(self.model_weights)

    def load_weights(self, model_weights):
        prefix = f'h.{self.layer_idx}.mlp'
        self.fc.weight.data.copy_(model_weights[f'{prefix}.c_fc.weight'].T)
        self.fc.bias.data.copy_(model_weights[f'{prefix}.c_fc.bias'])
        self.proj.weight.data.copy_(model_weights[f'{prefix}.c_proj.weight'].T)
        self.proj.bias.data.copy_(model_weights[f'{prefix}.c_proj.bias'])

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))

class MultiheadSelfAttention(nn.Module):
    def __init__(self, config, model_weights, layer_idx):
        super().__init__()
        self.config = config
        self.model_weights = model_weights
        self.layer_idx = layer_idx
        self.head_dim = self.config.n_embd // self.config.n_head

        self.qkv_proj = nn.Linear(self.config.n_embd, 3 * self.config.n_embd, bias=True)
        self.output_proj = nn.Linear(self.config.n_embd, self.config.n_embd, bias=True)
        self.load_weights(self.model_weights)

    def load_weights(self, model_weights):
        """ Load self-attention weights. """
        prefix = f'h.{self.layer_idx}.attn'
        self.qkv_proj.weight.data.copy_(model_weights[f'{prefix}.c_attn.weight'].T)
        self.qkv_proj.bias.data.copy_(model_weights[f'{prefix}.c_attn.bias'])
        self.output_proj.weight.data.copy_(model_weights[f'{prefix}.c_proj.weight'].T)
        self.output_proj.bias.data.copy_(model_weights[f'{prefix}.c_proj.bias'])

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(batch_size, seq_length, self.config.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.config.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.config.n_head, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq_length, seq_length)).to(x.device)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)

        return self.output_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config, model_weights, layer_idx):
        super().__init__()
        self.config = config
        self.model_weights = model_weights
        self.layer_idx = layer_idx

        # Layer Normalization before Attention
        self.norm1 = nn.LayerNorm(self.config.n_embd, eps=self.config.layer_norm_epsilon)
        self.attn = MultiheadSelfAttention(self.config, model_weights, layer_idx)

        # Layer Normalization before Feedforward Network
        self.norm2 = nn.LayerNorm(self.config.n_embd, eps=self.config.layer_norm_epsilon)
        self.mlp = FeedForwardNetwork(self.config, model_weights, layer_idx)

        self.load_weights(self.model_weights)

    def load_weights(self, model_weights):
        self.norm1.weight.data.copy_(self.model_weights[f'h.{self.layer_idx}.ln_1.weight'])
        self.norm1.bias.data.copy_(self.model_weights[f'h.{self.layer_idx}.ln_1.bias'])
        self.norm2.weight.data.copy_(self.model_weights[f'h.{self.layer_idx}.ln_2.weight'])
        self.norm2.bias.data.copy_(self.model_weights[f'h.{self.layer_idx}.ln_2.bias'])

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) # residual connection --> +
        x = x + self.mlp(self.norm2(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load model configuration and weights
        self.config = GPT2Config.from_pretrained(GPT2_CONFIG_PATH)
        self.model_weights = torch.load(GPT2_WEIGHTS_PATH, map_location='cpu')
        #print(self.model_weights.keys())

        # Initialize Embeddings
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embeddings = nn.Embedding(self.config.n_positions, self.config.n_embd)

        # Transformer Layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.config, self.model_weights, i) for i in range(self.config.n_layer)
        ])

        # Final Layer Norm
        self.final_norm = nn.LayerNorm(self.config.n_embd, eps=self.config.layer_norm_epsilon)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        
        #load weights
        self.load_weights(self.model_weights)

    def load_weights(self, model_weights):
        self.token_embeddings.weight.data.copy_(self.model_weights["wte.weight"])
        self.position_embeddings.weight.data.copy_(self.model_weights["wpe.weight"])
        self.final_norm.weight.data.copy_(self.model_weights['ln_f.weight'])
        self.final_norm.bias.data.copy_(self.model_weights['ln_f.bias'])
        # Tie weights
        self.lm_head.weight = self.token_embeddings.weight  

    def forward(self, x):
        
        #print("input_shape", x.shape) # input_shape torch.Size([1, 10])
        batch_size, seq_length = x.shape
        device = x.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        #print("position ids", position_ids.shape) # position ids torch.Size([1, 10])

        token_embedding = self.token_embeddings(x)
        #print("token_embedding", token_embedding.shape) # token_embedding torch.Size([1, 10, 768])
        #print("postion emb size", self.position_embeddings(position_ids).shape) #  postion embv size torch.Size([1, 10, 768])
        hidden_states = token_embedding + self.position_embeddings(position_ids)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
            #print("hidden_states", hidden_states.shape) # hidden_states torch.Size([1, 10, 768])

        hidden_states = self.final_norm(hidden_states)
        # print("hidden_states", hidden_states.shape) #  hidden_states torch.Size([1, 10, 768])
        logits = self.lm_head(hidden_states)
        #print("logits", logits.shape) # logits torch.Size([1, 10, 50257])
        return x