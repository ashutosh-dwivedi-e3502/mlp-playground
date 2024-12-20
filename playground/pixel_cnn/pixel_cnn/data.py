from datasets import load_dataset, DatasetDict
from jaxtyping import Float, Array
from typing import Tuple
from jax import numpy as jnp

def get_dataset() -> DatasetDict:
    print('loading dataset')    
    mnist = load_dataset("ylecun/mnist")
    # only select images because we don't need labels for this task
    mnist = mnist.with_format(type="jax")
    mnist = mnist.select_columns(column_names=["image"])
    # make the 28 X 28 image a 1 X 28 X 28 image (add channel dimension)

    def pre_process(batch_dict):
        batch = jnp.expand_dims(batch_dict['image'], axis=0)
        batch = (batch.astype(jnp.float32) / 255.0) * 2.0 - 1.0
        return {'image': batch}
   
    mnist = mnist.map(pre_process)
    print('done loading dataset')
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
    print('getting dataloader')
    mnist_dataset = get_dataset()
    dataloader = mnist_dataset['train'].iter(batch_size=batch_size)
    print('got dataloader')
    return dataloader