import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    obs_dim: int = 81
    act_dim: int = 5
    vocab_size: int = 1
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.pad_tensor_obs = nn.parameter.Parameter(
            data=torch.randn(1, 1, config.n_embd), requires_grad=True
        )
        self.pad_tensor_acts = nn.parameter.Parameter(
            data=torch.randn(1, 1, config.n_embd), requires_grad=True
        )
        self.pad_tensor_rews = nn.parameter.Parameter(
            data=torch.randn(1, 1, config.n_embd), requires_grad=True
        )

        self.transformer = nn.ModuleDict(dict(
            obs_embedding = nn.Embedding(config.obs_dim, config.n_embd),
            act_embedding = nn.Embedding(config.act_dim, config.n_embd),
            rew_embedding = nn.Embedding(2, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.act_dim, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, obs, act, rew):
        device = obs.device
        b, t = obs.shape[:2]
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        obs_emb = self.transformer.obs_embedding(obs).to(device)
        act_emb = self.transformer.act_embedding(act).to(device)
        rew_emb = self.transformer.rew_embedding(rew).to(device)
        
        pad_obs_batch = torch.tile(self.pad_tensor_obs, dims=(b, 1, 1)).to(device)
        pad_acts_batch = torch.tile(self.pad_tensor_acts, dims=(b, 1, 1)).to(device)
        pad_rews_batch = torch.tile(self.pad_tensor_rews, dims=(b, 1, 1)).to(device)
        
        obs_emb = torch.cat([pad_obs_batch, obs_emb], dim=1)
        act_emb = torch.cat([pad_acts_batch, act_emb], dim=1)
        rew_emb = torch.cat([pad_rews_batch, rew_emb], dim=1)
        
        sequence = self._stack_seq(obs_emb, act_emb, rew_emb).to(device)
        
        x = self.transformer.drop(sequence)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits[:, 1::3]

    
    def _stack_seq(self, obs, act, rew) -> torch.Tensor:
        _, timestep, _ = obs.shape

        stacked = []

        for i in range(timestep):
            stacked.append(obs[:, i:i+1, :])
            if i < act.shape[1]:
                stacked.append(act[:, i:i+1, :])
            if i < rew.shape[1]:
                stacked.append(rew[:, i:i+1, :])

        stacked = torch.cat(stacked, dim=1)

        return stacked

def compute_loss(x, y) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): action logits.
        y (torch.Tensor): the target action.
    Returns:
        float: the NLL loss.
    """
    assert y.dtype == torch.long
    assert x.shape[:-1] + (1,) == y.shape
    x = torch.nn.functional.log_softmax(x, dim=-1)
    return -torch.take_along_dim(x, y, dim=len(y.shape) - 1).sum(-1).mean()
