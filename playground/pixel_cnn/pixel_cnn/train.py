import equinox as eqx
import jax
import optax
import einops
import os

from jax import numpy as jnp
from jaxtyping import Array, Float, Scalar, PRNGKeyArray

from . import model, data, utils


def get_save_path():
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    return os.path.join(parent, 'saved_models/pixel_cnn.eqx')

@eqx.filter_value_and_grad(has_aux=True)
def compute_grads(model: model.PixelCNN, inputs: Float[Array, "batch channel height width"]):
    logits = jax.vmap(model)(inputs)
    loss = model.loss(logits, inputs)        
    accuracy = (logits.argmax(axis=-1) == inputs).mean()
    return jnp.mean(loss), accuracy


@eqx.filter_jit
def step_model(
    model: model.PixelCNN,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    images: Float[Array, "batch channel height width"],
):
    (loss, accuracy), grads = compute_grads(model, images)
    updates, new_state = optimizer.update(grads, state)

    model = eqx.apply_updates(model, updates)

    return model, new_state, (loss, accuracy)


def train(
    model: model.PixelCNN,
    dataloader,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    batch_size: int,
    num_epochs: int,
    print_every: int = 10,
):
    losses = []
    total_steps = 0

    for epoch in range(num_epochs):
        for step, batch_data  in enumerate(dataloader):            
            batch_data = batch_data['image'].astype(jnp.float32)

            # Split PRNG key for each step
            model, state, (loss, accuracy) = step_model(model, optimizer, state, batch_data)

            losses.append(loss)
            total_steps += 1

            if total_steps % print_every == 0:
                print(f"Epoch {epoch}, Step {step}: Loss {loss:.4f}, Accuracy {accuracy:.4f}")
                h_params = {'in_channels': model.in_channels, 'hidden_count': model.hidden_count}                
                utils.save_model(get_save_path(), h_params, model)

    return model, state, jnp.array(losses)
