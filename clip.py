import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    '''
    Each sentence has n_tokens tokens. Each token is embbeded to a n_embd vector.
    '''
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.positional_emb = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        # (B,seg_len) -> (B,seg_len,dim)
        x = self.token_embedding(tokens)

        x += self.positional_emb

        #(B,seg_len,dim)
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #(B, seq_len, dim)
        residue = x

        #self_Attn
        x = self.layernorm_1(x)
        x = self.attention(x, causual_mask = True)
        x += residue

        #feed_forward
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) #quickGULU activation
        x = self.linear_2(x)
        x += residue

        return x


         
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()

        ### fixed by SD, n_tokens = 77, n_embd = 768
        self.embedding = CLIPEmbedding(49408, 768, 77) 

        self.layers = nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (B,seq_len) --> (B,seg_len,dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        #(B,seg_len,dim)
        output = self.layernorm(state)

        return output