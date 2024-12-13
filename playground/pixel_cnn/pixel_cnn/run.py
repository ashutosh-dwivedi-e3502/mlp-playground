# %% [markdown]
# # Training a PixelCNN Model
# This notebook demonstrates training a PixelCNN model on MNIST data.

# %% Imports
import equinox as eqx
import jax
import optax
import math
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray

from . import model, train, data

# %% Define training function
def run_training(
    key: PRNGKeyArray,
    hidden_count: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 8,
    print_every: int = 10,
):
    """Run the PixelCNN training loop.
    
    Args:
        key: PRNG key for initialization
        hidden_count: Number of hidden channels in the model
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        batch_size: Batch size for training
        print_every: Print loss every N steps
    
    Returns:
        Trained model, optimizer state, and training losses
    """
    # Initialize model
    pixel_cnn = model.PixelCNN(
        key=key,
        in_channels=1,
        hidden_count=hidden_count
    )

    dataset = data.get_dataset()

    n_steps = math.ceil(dataset['train'].num_rows / batch_size)
    
    print(f'{n_steps=}')
    # Initialize optimizer
    lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=n_steps,
            decay_rate=0.99
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(eqx.filter(pixel_cnn, eqx.is_array))

    # Train model
    pixel_cnn, final_state, losses = train.train(
        model=pixel_cnn,
        dataloader=dataset['train'].iter(batch_size),
        optimizer=optimizer,
        state=opt_state,
        batch_size=batch_size,
        num_epochs=num_epochs,
        print_every=print_every
    )

    return pixel_cnn, final_state, losses

# %% Run training
key = jax.random.PRNGKey(0)
model, state, losses = run_training(key)
