# %%
from pixel_cnn.model import VerticalStackConvolution, HorizontalStackConv
from pixel_cnn.visualize import (
    visualize_receptive_field,
    compute_receptive_field_gradients,
)

import numpy as np
from jax import random
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
# %%
in_channels = 1
out_channels = 1
kernel_size = 3
mask_center = True
dilation = 1
key = random.PRNGKey(0)

# %%
# Visualize VerticalStackConvolution Receptive Field
print("Visualizing VerticalStackConvolution Receptive Field:")
vertical_conv = VerticalStackConvolution(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    mask_center=mask_center,
    dilation=dilation,
    key=key,
)
visualize_receptive_field(vertical_conv, input_shape=(in_channels, 5, 5))

# %%
# Visualize HorizontalStackConv Receptive Field
print("Visualizing HorizontalStackConv Receptive Field:")
horizontal_conv = HorizontalStackConv(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    mask_center=mask_center,
    dilation=dilation,
    key=key,
)
visualize_receptive_field(horizontal_conv, input_shape=(in_channels, 5, 5))

# %%
def func(input):
    h_output = horizontal_conv(input)
    v_output = vertical_conv(input)    
    return h_output + v_output

visualize_receptive_field(
    func,
    input_shape=(in_channels, 5, 5),
)

# Initialize random keys for parameter initialization
key = random.PRNGKey(0)
key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)

# Define the initial masked convolutions with mask_center=True
# These are used in the first layer
vert_conv = VerticalStackConvolution(
    in_channels=1, out_channels=1, kernel_size=3, mask_center=True, key=subkey1
)

horiz_conv = HorizontalStackConv(
    in_channels=1, out_channels=1, kernel_size=3, mask_center=True, key=subkey2
)

# Define the non-masked convolutions with mask_center=False
# These are reused in subsequent layers
vert_noc_conv = VerticalStackConvolution(
    in_channels=1, out_channels=1, kernel_size=3, mask_center=False, key=subkey3
)

horiz_noc_conv = HorizontalStackConv(
    in_channels=1, out_channels=1, kernel_size=3, mask_center=False, key=subkey4
)

# Define the function to build the network with multiple layers
def num_layer_network(inp, num_layers):
    # First layer with initial masked convolutions
    vert_img = vert_conv(inp)
    horiz_img = horiz_conv(inp) + vert_img
    # Subsequent layers with non-masked convolutions
    for _ in range(num_layers - 1):
        vert_img = vert_noc_conv(vert_img)
        horiz_img = horiz_noc_conv(horiz_img) + vert_img
    return horiz_img, vert_img

# Input image shape (in_channels, height, width)
input_shape = (1, 11, 11)
inp_img = jnp.zeros(input_shape, dtype=jnp.float32)

# Loop over different layer counts to visualize receptive field growth
for layer_count in range(2, 6):
    print(f"Layer {layer_count}")
    visualize_receptive_field(
        lambda inp: num_layer_network(inp, layer_count)[0], input_shape=input_shape
    )


# %%
