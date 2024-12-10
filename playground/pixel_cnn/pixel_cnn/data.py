from datasets import load_dataset, Dataset
from jaxtyping import Float, Array
from typing import Tuple
from jax import numpy as jnp

def get_dataset() -> Dataset:
    mnist = load_dataset("ylecun/mnist")
    # only select images because we don't need labels for this task
    mnist = mnist.with_format(type="jax")
    mnist = mnist.select_columns(column_names=["image"])
    # make the 28 X 28 image a 1 X 28 X 28 image (add channel dimension)    
    mnist = mnist.map(lambda x: {"image": jnp.expand_dims(x["image"], axis=0)})
    return mnist


def get_dataloader(    
    batch_size: int,
    shuffle: bool = True,
) -> Float[Array, "batch channel height width"]:
    """Get a dataloader for the dataset.
    
    Args:
        dataset: The dataset to load from
        batch_size: Size of each batch
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader that yields batches of shape (batch_size, channels, height, width)
    """
    mnist_dataset = get_dataset()    
    return mnist_dataset['train'].iter(batch_size=batch_size)