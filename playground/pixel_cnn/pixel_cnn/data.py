from datasets import load_dataset, Dataset
from jaxtyping import Float, Array
from typing import Tuple

def get_dataset(device: str) -> Dataset:
    mnist = load_dataset("ylecun/mnist")
    # only select images because we don't need labels for this task
    mnist = mnist.with_format(type="jax", columns=["image"], device=device)
    # make the 28 X 28 image a 1 X 28 X 28 image
    mnist = mnist.map(lambda x: {"image": x["image"].expand_dims(axis=0)}, batched=True)
    return mnist