import equinox as eqx
import jax
import optax
import torch 

from jax import numpy as jnp
from jax import random as jr

@eqx.filter_value_and_grad
def compute_grads(
    model: eqx.Module, images: jnp.ndarray, labels: jnp.ndarray, key
):
    logits = jax.vmap(model, in_axes=(0, None, 0))(images, True, key)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    return jnp.mean(loss)


@eqx.filter_jit
def step_model(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key,
):
    loss, grads = compute_grads(model, images, labels, key)
    updates, new_state = optimizer.update(grads, state, model)

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


def train(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    data_loader: torch.utils.data.DataLoader,
    batch_size:int,
    num_steps: int,
    print_every: int = 1000,
    key=None,
):
    losses = []

    def infinite_trainloader():
        while True:
            yield from data_loader

    for step, batch in zip(range(num_steps), infinite_trainloader()):
        images, labels = batch

        images = images.numpy()
        labels = labels.numpy()

        key, *subkeys = jr.split(key, num=batch_size + 1)
        subkeys = jnp.array(subkeys)

        (model, state, loss) = step_model(
            model, optimizer, state, images, labels, subkeys
        )

        losses.append(loss)

        if (step % print_every) == 0 or step == num_steps - 1:
            print(f"Step: {step}/{num_steps}, Loss: {loss}.")

    return model, state, losses