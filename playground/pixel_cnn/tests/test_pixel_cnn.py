from jax import random
import jax.numpy as jnp
import unittest

from pixel_cnn.model import VerticalStackConvolution, HorizontalStackConv
from pixel_cnn import utils
from functools import partial

def get_center_conv_output(conv_model, x):
    output = conv_model(x)
    return utils.get_center_pixel(output)


class TestVerticalConvReceptiveField(unittest.TestCase):
    def test_vertical_conv_receptive_field(self):
        # Set up test parameters
        in_channels = 1
        out_channels = 1
        kernel_size = 3
        mask_center = True
        dilation = 1
        
        key = random.PRNGKey(0)

        # Create vertical convolution layer
        vertical_conv = VerticalStackConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            mask_center=mask_center,
            dilation=dilation,
            key=key,
        )

        # Compute and check receptive field gradients
        image = jnp.zeros((1, 5, 5), dtype=jnp.float32)
        conv_output = vertical_conv(image)

        assert conv_output.shape == (1, 5, 5), f"Expected output shape (1,5,5) but got {conv_output.shape}"
        
        f = partial(get_center_conv_output, vertical_conv)
        gradients = utils.get_image_grad(f, image)
        
        # Verify gradients have expected shape
        assert gradients.shape == (5, 5), f"Expected gradient shape (5,5) but got {gradients.shape}"
        
        # Verify no gradients below center due to vertical masking
        center_row = gradients.shape[0] // 2
        assert jnp.all(gradients[center_row+1:] == 0), f"Found non-zero gradients below center row, {gradients=}"
        
        # Verify center pixel is masked
        assert gradients[center_row, center_row] == 0, "Center pixel should be masked"


    def test_horizontal_conv_receptive_field(self):
        in_channels = 1
        out_channels = 1
        kernel_size = 3
        mask_center = True
        dilation = 1

        key = random.PRNGKey(0)

        horizontal_conv = HorizontalStackConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            mask_center=mask_center,
            dilation=dilation,
            key=key,
        )

                # Compute and check receptive field gradients
        image = jnp.zeros((1, 5, 5), dtype=jnp.float32)
        conv_output = horizontal_conv(image)

        assert conv_output.shape == (1, 5, 5), f"Expected output shape (1,5,5) but got {conv_output.shape}"
        
        f = partial(get_center_conv_output, horizontal_conv)
        gradients = utils.get_image_grad(f, image)

        assert gradients.shape == (5, 5), f"Expected gradient shape (5,5) but got {gradients.shape}"
        center_col = gradients.shape[1] // 2
        assert jnp.all(gradients[:, center_col+1:] == 0), f"Found non-zero gradients to the right of center column, {gradients=}"
        assert gradients[center_col, center_col] == 0, "Center pixel should be masked"

if __name__ == '__main__':
    unittest.main()