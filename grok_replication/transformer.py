# mostly copied from [OpenAI's implementation](https://github.com/openai/grok/blob/main/grok/transformer.py)

from typing import Tuple, List, Union
import torch
import torch.nn as nn
from numpy import cos, sin, sqrt
from torch import tensor, Tensor
from sklearn.decomposition import PCA

class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int) -> None:
        super().__init__()

        self.d_key = d_key

        # head projections
        self.Wq = nn.Linear(d_model, d_key, bias=False, dtype=torch.float64)
        self.Wk = nn.Linear(d_model, d_key, bias=False, dtype=torch.float64)
        self.Wv = nn.Linear(d_model, d_key, bias=False, dtype=torch.float64)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Union[Tensor, None] = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:

        # project queries, keys, values
        queries = self.Wq(queries)
        keys = self.Wk(keys)
        values = self.Wv(values)

        # calculate compatibility function
        attn = torch.matmul(queries, torch.transpose(keys, -2, -1))
        attn = attn / sqrt(self.d_key)

        # Filter out attention to future positions
        if mask is not None:
            attn.masked_fill_(mask == 0, float("-inf"))

        # softmax
        attn = self.softmax(attn) 

        # sum the weighted value vectors
        result: Tensor = torch.matmul(attn, values)  # shape = (max_context_len, d_key)
        if save_activations:
            leaf_attn = attn.clone().detach()  # type: ignore
            leaf_values = values.clone().detach()  # type: ignore
        else:
            leaf_attn = None  # type: ignore
            leaf_values = None  # type: ignore

        return result, leaf_attn, leaf_values


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        d_key = int(d_model / heads)

        attn_heads = [
            AttentionHead(d_model, d_key)
            for _ in range(heads)
        ]
        self.attn_heads = nn.ModuleList(attn_heads)
        self.Wo = nn.Linear(d_model, d_model, bias=False, dtype=torch.float64)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor = None,
        save_activations=False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:

        head_outputs = [
            h(
                queries=queries,
                keys=keys,
                values=values,
                mask=mask,
                save_activations=save_activations,
            )
            for h in self.attn_heads
        ]
        head_results = [output[0] for output in head_outputs]

        if save_activations:
            layer_attns = list([output[1] for output in head_outputs])
            layer_values = list([output[2] for output in head_outputs])
        else:
            layer_attns = []
            layer_values = []

        multihead_result = torch.cat(head_results, dim=-1)
        multihead_result = self.Wo(multihead_result)
        return multihead_result, layer_attns, layer_values


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        multiplier: int = 4,
        non_linearity: str = "relu",
    ) -> None:
        super().__init__()

        d_ff = int(multiplier * d_model)

        non_linearities = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky": nn.LeakyReLU}

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            non_linearities[non_linearity](),
            nn.Linear(d_ff, d_model, bias=False),
        ).to(torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float,
        non_linearity: str = "relu",
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, heads)
        self.self_attn_norm = nn.LayerNorm(d_model, dtype=torch.float64) # TODO: can I remove this?

        self.ffn = FFN(d_model, non_linearity=non_linearity)
        self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = nn.LayerNorm(d_model, dtype=torch.float64) # TODO: can I remove this?

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:

        a1, layer_attns, layer_values = self.self_attn(
            x, x, x, self_attn_mask, save_activations
        )

        a1 = self.self_attn_norm(x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a2 = self.ffn_norm(a1 + a2)
        return a2, layer_attns, layer_values


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        num_blocks: int,
        dropout: float,
        non_linearity: str = "relu",
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, heads, dropout, non_linearity)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations=False,
    ) -> Tuple[Tensor, List[List[Tensor]], List[List[Tensor]]]:

        a = x
        attentions = []
        values = []
        for block in self.blocks:
            a, layer_attentions, layer_values = block(
                a, self_attn_mask, save_activations=save_activations
            )
            if save_activations:
                attentions.append(layer_attentions)
                values.append(layer_values)
        return a, attentions, values


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        max_context_len: int = 1024,
        vocab_len: int = 2000,
        non_linearity: str = "relu",
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.max_context_len = max_context_len
        self.non_linearity = non_linearity
        self.embedding_on = True

        self.vocab_len = vocab_len

        self.embedding = nn.Embedding(vocab_len, d_model) 
        self.register_buffer(
            "position_encoding", self._position_encoding(max_context_len, d_model)
        )
        self.register_buffer("self_attn_mask", self.make_mask(max_context_len))

        self.decoder = Decoder(
            d_model,
            n_heads,
            n_layers,
            dropout,
            self.non_linearity,
        )

        self.linear = nn.Linear(d_model, vocab_len, bias=False, dtype=torch.float64)

    def use_embed(self, use_embed: bool) -> None:
        self.embedding_on = use_embed

    def get_low_d_transform(self, dataset, n_features):
        # dataset is n x d
        # use PCA to reduce dimensionality to n_features
        import numpy as np
        embedded_dataset = self.embed(dataset).cpu().numpy()
        embedded_dataset = np.reshape(embedded_dataset, (len(dataset), -1))
        pca = PCA(n_components=n_features)
        pca.fit(embedded_dataset)
        def transform(x):
            embed_x = self.embed(x.cuda()).cpu().numpy()
            shape_embed = embed_x.shape
            embed_x = np.reshape(embed_x, (len(x), -1))
            return torch.from_numpy(pca.inverse_transform(pca.transform(embed_x)).reshape(shape_embed))
        return transform


    @staticmethod
    def make_mask(context_len: int) -> Tensor:
        return torch.ones([context_len, context_len]).tril()

    @classmethod
    def _position_encoding(cls, context_len: int, d_model: int) -> Tensor:
        rows = [
            tensor([
                sin(pos / (10000 ** (i / d_model)))
                if i % 2 == 0
                else cos(pos / (10000 ** ((i - 1) / d_model)))
                for i in range(d_model)
            ])
            for pos in range(context_len)
        ]
        stack = torch.stack(rows, dim=1)
        return stack.T

    def embed(self, indices: Tensor) -> Tensor:
        context_len = indices.shape[-1]
        pe = self.position_encoding[:context_len, :]  # type: ignore
        embedded = self.embedding(indices)
        return pe + embedded

    def forward(
        self,
        x: Tensor,
        pos: int = None,
        save_activations: bool = False,
        embedding_noise: float = 0.0,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        """parameters:
        x:  (rank-1 tensor) vocab indices of decoder input token
                     sequence"""

        # Make sure sampling inputs are on the correct device
        x = x.to(self.embedding.weight.device)

        # make_attention mask
        if self.embedding_on:
            this_max_context_len = x.shape[-1]
            self_attn_mask = self.self_attn_mask[  # type: ignore
                :this_max_context_len, :this_max_context_len
            ]
            x = self.embed(x)
        else:
            this_max_context_len = x.shape[-2]
            self_attn_mask = self.self_attn_mask[  # type: ignore
                :this_max_context_len, :this_max_context_len
            ]

        if embedding_noise > 0.0:
            x = x + torch.randn_like(x) * embedding_noise

        decoded, attentions, values = self.decoder(
            x, self_attn_mask, save_activations=save_activations
        )

        # Return predictions for specific token
        if pos is not None:
            decoded = decoded[:, pos, :]

        y_hat = self.linear(decoded)
        return y_hat