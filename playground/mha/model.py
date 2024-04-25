import equinox as eqx
import einops
import jax
import math

from equinox import nn
from jax import random as jr
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    # attn_logits = jnp.matmul(q, einops.rearrange(k, "... seq_len dims -> ... dims seq_len"))
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
    embed_dim: int
    num_heads: int
    qkv_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear

    def __init__(self, embed_dim: int, num_heads: int, key:PRNGKeyArray):
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        key_qkv, key_proj = jr.split(key, 2)

        # TODO: is bias initialization and kernel init with xavier important?
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

    def __call__(self, x: Float[Array, "batch_size seq_len embed_dim"], mask=None):
        batch_size, seq_len, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)

        print(f"{batch_size=}")
        print(f"{seq_len=}")

        # a single projection layer, given the input produces Q, K, V matrices
        reshaped_x = einops.rearrange(
            x, "batch_size seq_len embedding_dim -> (batch_size seq_len) embedding_dim"
        )
        qkv = jax.vmap(self.qkv_proj)(reshaped_x)

        # The scaled dot product attention allows a network to attend over a sequence.
        # However, often there are multiple different aspects a sequence element
        # wants to attend to, and a single weighted average is not a good option for it.
        # This is why we extend the attention mechanisms to multiple heads,
        # i.e. multiple different query-key-value triplets on the same features.
        # Specifically, given a query, key, and value matrix, we transform those into sub-queries,
        # sub-keys, and sub-values, which we pass through the scaled dot product attention independently.
        # Afterward, we concatenate the heads and combine them with a final weight matrix

        reshaped_qkv = einops.rearrange(
            qkv,
            "(b s) (h d) -> b h s d",
            s=seq_len,
            b=batch_size,
            h=self.num_heads,
        )
        # embedding dims contains all of qkv, so split
        q, k, v = jnp.array_split(reshaped_qkv, 3, axis=-1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        output_embeddings = jax.vmap(self.output_proj)(
            einops.rearrange(values, "b h s d -> (b s) (h d)")
        )
        # combine the heads dim seperate out seq len and batch dim
        output_embeddings = einops.rearrange(
            output_embeddings,
            "(b s) d -> b s d",
            s=seq_len,
            b=batch_size
        )
        return output_embeddings, attention


class EncoderBlock(eqx.Module):
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

    def __init__(self, input_dim:int, num_heads: int, dim_feedforward: int, dropout_prob: float, key:PRNGKeyArray):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob

        key_attn, key_lin1, key_lin2 = jr.split(key, 3)

        self.self_attn = MultiHeadAttention(embed_dim=self.input_dim, num_heads=self.num_heads, key=key_attn)

        self.linear1 = nn.Linear(in_features=self.input_dim, out_features=self.dim_feedforward, use_bias=True, key=key_lin1)
        self.linear2 = nn.Linear(in_features=self.dim_feedforward, out_features=self.input_dim, use_bias=True, key=key_lin2)
        self.norm1 = nn.LayerNorm(shape=self.dim_feedforward)
        self.norm2 = nn.LayerNorm(shape=self.input_dim)
        self.dropout = nn.Dropout(p=dropout_prob)


    def __call__(self, x, key:PRNGKeyArray, mask=None, train=True):
        dropout_key = jr.split(key, 1)

        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, inference=not train, key=dropout_key)
        x = jax.vmap(self.norm1)(x)

        mlp_out = x # keep a copy of x for residual connection
        mlp_out = jax.vmap(self.linear1)(mlp_out)
        mlp_out = self.dropout(mlp_out, key=dropout_key)
        mlp_out = jax.nn.relu(mlp_out)
        mlp_out = jax.vmap(self.linear2)(mlp_out)

        x = x + self.dropout(mlp_out, inference=not train, key=dropout_key)
        x = jax.vmap(self.norm2)(x)
        return x
