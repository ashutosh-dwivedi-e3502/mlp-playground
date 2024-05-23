from jax import numpy as jnp
from jax import random as jr
from jax import nn
from equinox import nn as enn

import model
import dataloader
import train
# from . import model, dataloader, train

import equinox as eqx
import jax
import math
import optax


model_dim = 32
num_heads = 1
num_layers = 1
dropout_prob = 0.0
lr = 5e-4
warmup = 50


(
    train_dataloader,
    test_dataloader,
    val_dataloader,
) = dataloader.get_reversed_data_loaders()


train_row = next(iter(train_dataloader))
print(jnp.array(train_row).shape)
print(f"{train_dataloader.batch_size=}")


key = jr.PRNGKey(2003)

num_steps = 10000


model_obj = model.TransformerPredictor(
    num_layers=5,
    model_dim=128,
    num_classes=9,
    num_heads=1,
    dropout_prob=0.15,
    input_dropout_prob=0.05,
    key=key,
)


optimizer = optax.adamw(learning_rate=lr)

state = optimizer.init(eqx.filter(model_obj, eqx.is_inexact_array))

model_obj, state, losses = train.train(
    model_obj,
    optimizer,
    state,
    train_dataloader,
    train_dataloader.batch_size,
    num_steps,
    key=key,
)
