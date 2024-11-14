import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random


def plot_receptive_field(img_grads):
    """
    Plots a visualization of the receptive field based on gradient information.
    
    Creates two side-by-side plots:
    1. A weighted receptive field visualization showing gradient magnitudes using a viridis colormap
    2. A binary receptive field visualization showing which pixels have non-zero gradients
    
    If the center pixel has no gradient influence (zero value), it is marked in red.
    
    Args:
        img_grads: numpy array of shape (height, width) containing the gradient values
                  showing each pixel's influence on the center output pixel.
                  Higher values indicate stronger influence.
    """
    # Plot receptive field
    fig, ax = plt.subplots(1, 2)
    pos = ax[0].imshow(img_grads)
    fig.colorbar(pos, ax=ax[0])
    ax[1].imshow(img_grads > 0)

    # Mark the center pixel in red if it doesn't have any gradients
    h_center = img_grads.shape[0] // 2
    w_center = img_grads.shape[1] // 2
    show_center = (img_grads[h_center, w_center] == 0)
    if show_center:
        center_pixel = np.zeros(img_grads.shape + (4,))
        center_pixel[h_center, w_center, :] = np.array([1.0, 0.0, 0.0, 1.0])
    for i in range(2):
        ax[i].axis('off')
        if show_center:
            ax[i].imshow(center_pixel)
    ax[0].set_title("Weighted Receptive Field")
    ax[1].set_title("Binary Receptive Field")
    plt.show()
    plt.close()

def compute_receptive_field_gradients(model, input_shape):
    """
    Computes gradients showing the receptive field of the center pixel.
    Args:
        model: The convolutional model to analyze.
        input_shape: The shape of the input image (in_channels, height, width).
    Returns:
        The gradient array showing receptive field influence.
    """
    # Create an input image filled with zeros
    img = jnp.zeros(input_shape, dtype=jnp.float32)

    # Function to apply the model and get the output value at the center pixel
    def apply_fn(x):
        output = model(x)
        # Assuming output shape is (out_channels, height, width)
        # Sum over channels to get a scalar output
        h_center = output.shape[1] // 2
        w_center = output.shape[2] // 2
        return output[:, h_center, w_center].sum()

    # Compute gradients of the output center pixel w.r.t the input
    print(f"{img.shape=}")
    grad_fn = jax.grad(apply_fn)
    img_grads = jnp.abs(grad_fn(img))
    print(f"{img_grads.shape=}")
    img_grads = jax.device_get(img_grads)

    # Since img_grads has shape (in_channels, height, width), sum over channels
    img_grads = img_grads.sum(axis=0)  # Now shape is (height, width)    
    print(f"After summing over channels: {img_grads.shape=}")
    
    return img_grads
    
def visualize_receptive_field(model, input_shape):
    """
    Visualizes the receptive field of the center pixel in the output of the model.
    Args:
        model: The convolutional model to analyze.
        input_shape: The shape of the input image (in_channels, height, width).
    """
    img_grads = compute_receptive_field_gradients(model, input_shape)
    plot_receptive_field(img_grads)
