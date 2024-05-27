import equinox as eqx
import einops
import functools
import jax
import math

from equinox import nn
from jax import random as jr
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import List, Union, Callable


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(
        q, einops.rearrange(k, "... seq_len dims -> ... dims seq_len")
    )
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = jax.nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


def expand_mask(mask):
    assert mask.ndim > 2, "mask must be atleast 2 dim"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiHeadAttention(eqx.Module):
    """Given initial embeddings, get q k v, apply attention, and output projection"""

    embed_dim: int
    num_heads: int
    qkv_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear

    def __init__(self, embed_dim: int, num_heads: int, key: PRNGKeyArray):
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        key_qkv, key_proj = jr.split(key, 2)

        # TODO: is bias initialization and kernel init with xavier required?
        qkv_out_size = 3 * self.embed_dim
        self.qkv_proj = nn.Linear(
            in_features=self.embed_dim,
            out_features=qkv_out_size,
            key=key_qkv,
            use_bias=True,
        )
        self.output_proj = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            use_bias=True,
            key=key_proj,
        )

    def __call__(self, x: Float[Array, "seq_len embed_dim"], mask=None):
        seq_len, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)

        # a single projection layer, given the input produces Q, K, V matrices
        qkv = jax.vmap(self.qkv_proj)(x)

        # The scaled dot product attention allows a network to attend over a sequence.
        # However, often there are multiple different aspects a sequence element
        # wants to attend to, and a single weighted average is not a good option for it.
        # This is why we extend the attention mechanisms to multiple heads,
        # i.e. multiple different query-key-value triplets on the same features.
        # Specifically, given a query, key, and value matrix, we transform those into sub-queries,
        # sub-keys, and sub-values, which we pass through the scaled dot product attention independently.
        # Afterward, we concatenate the heads and combine them with a final weight matrix

        # split the embeding_dim into multiple heads
        # dim here is different from embed_dim, it's 3 * embed_dims
        reshaped_qkv = einops.rearrange(
            qkv,
            "seq_len (num_heads d) -> num_heads seq_len d",
            seq_len=seq_len,
            num_heads=self.num_heads,
        )
        # embedding dims contains all of qkv, so split
        q, k, v = jnp.array_split(reshaped_qkv, 3, axis=-1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = einops.rearrange(
            values,
            "num_heads seq_len d -> seq_len (num_heads d)",
            num_heads=self.num_heads,
            seq_len=seq_len,
        )
        output_embeddings = jax.vmap(self.output_proj)(values)
        return output_embeddings, attention


class EncoderBlock(eqx.Module):
    """
    This FF block, which accepts the embeddings and returns the output
    after passing them through a linear layers with dropouts
    """

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    self_attn: MultiHeadAttention

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout_prob: float,
        key: PRNGKeyArray,
    ):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob

        key_attn, key_lin1, key_lin2 = jr.split(key, 3)

        self.norm1 = nn.LayerNorm(self.input_dim)

        self.self_attn = MultiHeadAttention(
            embed_dim=self.input_dim, num_heads=self.num_heads, key=key_attn
        )

        self.linear1 = nn.Linear(
            in_features=self.input_dim,
            out_features=self.dim_feedforward,
            use_bias=True,
            key=key_lin1,
        )
        self.linear2 = nn.Linear(
            in_features=self.dim_feedforward,
            out_features=self.input_dim,
            use_bias=True,
            key=key_lin2,
        )
        self.norm2 = nn.LayerNorm(shape=self.input_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def __call__(
        self,
        x: Float[Array, "seq_len embed_dim"],
        key: PRNGKeyArray,
        mask=None,
        train=True,
    ):
        dropout_key = jr.split(key, 1)

        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, inference=not train, key=dropout_key)
        x = jax.vmap(self.norm1)(x)

        mlp_out = x  # keep a copy of x for residual connection
        mlp_out = jax.vmap(self.linear1)(mlp_out)
        mlp_out = self.dropout(mlp_out, key=dropout_key)
        mlp_out = jax.nn.relu(mlp_out)
        mlp_out = jax.vmap(self.linear2)(mlp_out)

        x = x + self.dropout(mlp_out, inference=not train, key=dropout_key)
        x = jax.vmap(self.norm2)(x)
        return x


class TransformerEncoder(eqx.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float
    encoders: List[EncoderBlock]

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout_prob,
        key: PRNGKeyArray,
    ):
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob
        self.encoders = [
            EncoderBlock(input_dim, num_heads, dim_feedforward, dropout_prob, key)
            for _ in range(num_layers)
        ]

    def __call__(
        self, x: Float[Array, "seq_len dim"], key: PRNGKeyArray, mask=None, train=True
    ):
        for l in self.encoders:
            layer_partial = functools.partial(l, key=key, mask=mask, train=train)
            x = layer_partial(x)
        return x

    def get_attention_maps(self, x, mask=None, train=True):
        # A function to return the attention maps within the model for a single application
        # Used for visualization purpose later
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask)
            attention_maps.append(attn_map)
            x = l(x, mask=mask, train=train)
        return attention_maps


class PositionalEncoding(eqx.Module):
    model_dim: int  # Hidden dimensionality of the input.
    max_len: int = 5000  # Maximum length of a sequence to expect.

    pe: Float[Array, "max_len model_dim"]

    def __init__(self, model_dim: int, max_len: int = 5000):
        """Constructs positional encoding for transformer models.

        Args:
            max_len: Maximum sequence length.
            d_model: Embedding dimension.

        Returns:
            A Jax numpy array of shape (1, max_len, d_model).
        """
        self.model_dim = model_dim
        self.max_len = max_len

        pe = jnp.zeros((self.max_len, self.model_dim))

        position = jnp.arange(0, self.max_len, dtype=jnp.float32)  # shape: (max_len,)
        position = jnp.expand_dims(position, axis=1)  # shape: (max_len, 1)

        div_term = jnp.exp(
            jnp.arange(0, self.model_dim, 2) * (-math.log(10000.0) / self.model_dim)
        )  # shape: (model_dim/2,)

        even_idxs = jnp.arange(0, self.model_dim, 2)
        odd_idxs = jnp.arange(0, self.model_dim, 2)
        pe.at[:, even_idxs].set(jnp.sin(position * div_term))
        pe.at[:, odd_idxs].set(jnp.cos(position * div_term))

        self.pe = pe

    def __call__(self, x: Float[Array, "seq_len model_dim"]):
        seq_len = x.shape[0]
        x = x + self.pe[:seq_len]
        return x


class TransformerPredictor(eqx.Module):
    num_layers: int
    model_dim: int
    num_classes: int
    num_heads: int
    dropout_prob: float = 0.0
    input_dropout_prob: float = 0.0

    input_dropout: nn.Dropout
    input_layer: nn.Linear
    positional_encoding: PositionalEncoding
    transformer: TransformerEncoder
    output_net: List[Union[Callable, eqx.Module]]

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        num_classes: int,
        num_heads: int,
        dropout_prob: float,
        input_dropout_prob: float,
        key=PRNGKeyArray,
    ):
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.input_dropout_prob = input_dropout_prob

        input_layer_key, transformer_encoder_key, output_net_key_1, output_net_key_2 = (
            jr.split(key, 4)
        )

        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer = nn.Linear(
            in_features=self.num_classes, out_features=self.model_dim, key=input_layer_key
        )
        self.positional_encoding = PositionalEncoding(self.model_dim)

        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=self.model_dim,
            dim_feedforward=2 * self.model_dim,
            num_heads=self.num_heads,
            dropout_prob=self.dropout_prob,
            key=transformer_encoder_key,
        )

        self.output_net = [
            nn.Linear(
                in_features=self.model_dim,
                out_features=self.model_dim,
                key=output_net_key_1,
            ),
            nn.LayerNorm(self.model_dim),
            jax.nn.relu,
            nn.Dropout(self.dropout_prob),
            nn.Linear(
                in_features=self.model_dim,
                out_features=self.model_dim,
                key=output_net_key_2,
            ),
        ]

    def __call__(
        self,
        x: Float[Array, "seq_len"],
        mask=None,
        add_positional_encoding=True,
        train=True,
        key: PRNGKeyArray = None,
    ):

        output_key, transformer_key, dropout_key = (None, None) if key is None else jax.random.split(key, 3)

        x = self.input_dropout(x, deterministic=not train, key=dropout_key)
        x = jax.vmap(self.input_layer)(x)

        if add_positional_encoding:
            x = self.positional_encoding(x)

        x = self.transformer(x, mask=mask, train=train, key=transformer_key)

        for l in self.output_net:
            x = (
                jax.vmap(l)(x)
                if not isinstance(l, nn.Dropout)
                else l(x, deterministic=not train, key=output_key)
            )
        return x

    def get_attention_maps(
        self, x, mask=None, add_positional_encodding=True, train=True
    ):
        x = self.input_dropout(x, deterministic=not train)
        x = self.input_layer(x)
        if add_positional_encodding:
            x = self.positional_encoding(x)
        attention_maps = self.tranformer.get_attention_maps(x, mask=mask, train=train)
        return attention_maps
