import equinox as eqx
import functools
import jax
import optax
import torch

from jax import numpy as jnp
from jax import random as jr


@eqx.filter_value_and_grad(has_aux=True)
def compute_grads(model: eqx.Module, inputs: jnp.ndarray, labels: jnp.ndarray, keys):
    logits = jax.vmap(model, in_axes=(0, None, None, None, 0))(
        inputs, None, True, True, keys
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    acc = (logits.argmax(axis=-1) == labels).mean()
    return jnp.mean(loss), acc


@eqx.filter_jit
def step_model(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key,
):
    (loss, acc), grads = compute_grads(model, images, labels, key)
    updates, new_state = optimizer.update(grads, state, model)

    model = eqx.apply_updates(model, updates)

    return model, new_state, (loss, acc)


def train(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    data_loader: torch.utils.data.DataLoader,
    batch_size: int,
    num_steps: int,
    print_every: int = 1000,
    key=None,
):
    losses = []

    def infinite_trainloader():
        while True:
            yield from data_loader

    for step, batch in zip(range(num_steps), infinite_trainloader()):
        data, labels = batch

        data_one_hot = jax.nn.one_hot(data, num_classes=model.num_classes)

        batch_size = data.shape[0]

        key, *subkeys = jr.split(key, num=batch_size + 1)
        subkeys = jnp.array(subkeys)

        (model, state, (loss, acc) ) = step_model(
            model, optimizer, state, data_one_hot, labels, subkeys
        )

        losses.append(loss)

        if (step % print_every) == 0 or step == num_steps - 1:
            print(f"Step: {step}/{num_steps}, {loss=}, {acc=}")

    return model, state, losses
