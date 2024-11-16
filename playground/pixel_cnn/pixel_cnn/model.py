import equinox as eqx
import jax
import jax.numpy as jnp

from typing import Optional
from jaxtyping import Array, Float

class MaskedConv(eqx.Module):
    """A masked convolution module using Equinox"""
    
    in_channels: int
    out_channels: int
    mask: Float[Array, "kernel_h kernel_w"]  
    dilation: int = 1 
    conv: eqx.nn.Conv2d
    key: Optional[jax.random.PRNGKey] = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask: Float[Array, "kernel_h kernel_w"],
        dilation: int = 1,
        key: Optional[jax.random.PRNGKey] = None
    ) -> None:
        # Set the output channels, input channels, and dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.key = key

        # Ensure mask is a JAX array had has right dims
        self.mask = jnp.array(mask)
        assert self.mask.ndim == 2, "Mask must be a 2D array."

        # Initialize the convolution layer
        kernel_height, kernel_width = self.mask.shape

        pad = kernel_height // 2
        padding=((pad, pad), (pad, pad))

        self.conv = eqx.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(kernel_height, kernel_width),
            stride=1,
            padding=padding,
            use_bias=True,
            dilation=self.dilation,
            key=self.key
        )
        self.mask = self.mask.reshape(1, 1, kernel_height, kernel_width)

    def __call__(self, x: Float[Array, "batch h w c_in"]) -> Float[Array, "batch h w c_out"]:
        masked_weights = self.conv.weight * self.mask
        masked_conv = eqx.tree_at(
            where=lambda conv: conv.weight,
            pytree=self.conv,
            replace=masked_weights
        )
        return masked_conv(x)


class VerticalStackConvolution(eqx.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    mask_center: bool = False
    dilation: int = 1
    conv: MaskedConv

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask_center: bool = False,
        dilation: int = 1,
        key: Optional[jax.random.PRNGKey] = None
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_center = mask_center
        self.dilation = dilation

        # Create the mask
        self.kernel_size = kernel_size
        mask = jnp.ones((self.kernel_size, self.kernel_size), dtype=jnp.float32)
        # Mask out all pixels below the center row
        mask = mask.at[self.kernel_size // 2 + 1:, :].set(0)
        # Optionally mask out the center row
        if self.mask_center:
            mask = mask.at[self.kernel_size // 2, :].set(0)
                
        # Initialize the MaskedConv module
        self.conv = MaskedConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            mask=mask,
            dilation=self.dilation,
            key=key
        )

    def __call__(self, x: Float[Array, "c_in h w"]) -> Float[Array, "c_out h w"]:
        return self.conv(x)


class HorizontalStackConv(eqx.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    mask_center: bool = False
    dilation: int = 1
    conv: MaskedConv

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 mask_center: bool = False, dilation: int = 1, key=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mask_center = mask_center
        self.dilation = dilation

        # Create the mask using jnp instead of np
        mask = jnp.ones((1, self.kernel_size), dtype=jnp.float32)
        mask = mask.at[0, self.kernel_size // 2 + 1:].set(0)
        # For the first convolution, we will also mask the center pixel
        if self.mask_center:
            mask = mask.at[0, self.kernel_size // 2].set(0)

        # Initialize the MaskedConv module
        self.conv = MaskedConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            mask=mask,
            dilation=self.dilation,
            key=key
        )

    def __call__(self, x):
        return self.conv(x)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import random

    input_shape = (7, 11, 11)
    key = random.PRNGKey(0)
    img = jnp.zeros(input_shape, dtype=jnp.float32)
    vertical_model = VerticalStackConvolution(in_channels=input_shape[0], out_channels=13, kernel_size=7, mask_center=True, key=key)
    vertical_model(img)
    print('vertical worked')
    horizontal_model = HorizontalStackConv(in_channels=input_shape[0], out_channels=13, kernel_size=7, mask_center=True, key=key)
    horizontal_model(img)