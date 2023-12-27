import einops
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import matplotlib.pyplot as plt


def img_to_patches(
    image: Float[Array, "channel height width"], patch_size: int, flatten_channel: bool
) -> Float[Array, "num_patches flattened_patch_dim"]:
    # hx = h / ph, wx = w / pw

    if flatten_channel:
        rearrange_str = "c (hx ph) (wx pw) -> (hx wx) (c ph pw)"
    else:
        rearrange_str = "c (hx ph) (wx pw) -> (hx wx) c ph pw"

    return einops.rearrange(
        image,
        rearrange_str,
        ph=patch_size,
        pw=patch_size,
    )


def plot_image(image):
    image = einops.rearrange(image, "c h w) -> (h w c")
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def plot_patches(patches):
    num_patches = patches.shape[0]
    fig, axes = plt.subplots(1, num_patches, figsize=(num_patches * 5, 5))

    for i in range(num_patches):
        patch = patches[i]
        patch = einops.rearrange(patch, "c h w -> h w c")
        axes[i].imshow(patch)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
