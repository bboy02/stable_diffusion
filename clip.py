import torch
from torch import nn 
from torch.nn import functional as F 
from attention import SelfAttention


class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embd: int , n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):

        # (Batch_Size, Seq_len) -> (Batch_Size, Seq_len, Dim)

        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x
    

class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd , n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (Batch_Size, Seq_Len, Dim)

        residue = x

        ## Self attention

        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=True)

        x += residue

        ## feedforward layer

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) #quick GELU
        
        x = self.linear_2(x)
        x += residue

        return x





class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768 ,77)
        # clipembedding (vocab_size, embed_size, sequence_len)
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)]
        )
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long)


        state = self.embedding(tokens)
        # (Batch_Size, Seq_len) -> (Batch_size ,Seq_len, Embed_Dim)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output

        

