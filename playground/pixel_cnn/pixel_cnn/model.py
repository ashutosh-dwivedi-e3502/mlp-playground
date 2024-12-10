import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import einops

from typing import List, Optional, Tuple
from jaxtyping import Array, Float, Scalar, PRNGKeyArray


class MaskedConv(eqx.Module):
    """A masked convolution module using Equinox"""

    conv: eqx.nn.Conv2d
    key: PRNGKeyArray
    in_channels: int
    out_channels: int
    mask: Float[Array, "kernel_h kernel_w"]
    dilation: int = 1

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        mask: Float[Array, "kernel_h kernel_w"],
        dilation: int = 1,
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

        pad_h, pad_w = (kernel_height - 1) * dilation // 2, (
            kernel_width - 1
        ) * dilation // 2
        padding = ((pad_h, pad_h), (pad_w, pad_w))

        self.conv = eqx.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(kernel_height, kernel_width),
            stride=1,
            padding=padding,
            use_bias=True,
            dilation=self.dilation,
            key=self.key,
        )
        self.mask = self.mask.reshape(1, 1, kernel_height, kernel_width)

    def __call__(
        self, x: Float[Array, "in_channel height width"]
    ) -> Float[Array, "out_channel height width"]:
        masked_weights = self.conv.weight * self.mask
        masked_conv = eqx.tree_at(
            where=lambda conv: conv.weight, pytree=self.conv, replace=masked_weights
        )
        return masked_conv(x)


class VerticalStackConv(eqx.Module):
    conv: MaskedConv
    in_channels: int
    out_channels: int
    kernel_size: int
    mask_center: bool = False
    dilation: int = 1

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask_center: bool = False,
        dilation: int = 1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_center = mask_center
        self.dilation = dilation

        # Create the mask
        self.kernel_size = kernel_size
        mask = jnp.ones((self.kernel_size, self.kernel_size), dtype=jnp.float32)
        # Mask out all pixels below the center row
        mask = mask.at[self.kernel_size // 2 + 1 :, :].set(0)
        # Optionally mask out the center row
        if self.mask_center:
            mask = mask.at[self.kernel_size // 2, :].set(0)

        # Initialize the MaskedConv module
        self.conv = MaskedConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            mask=mask,
            dilation=self.dilation,
            key=key,
        )

    def __call__(
        self, x: Float[Array, "in_channel height width"]
    ) -> Float[Array, "out_channel height width"]:
        return self.conv(x)


class HorizontalStackConv(eqx.Module):
    conv: MaskedConv
    in_channels: int
    out_channels: int
    kernel_size: int
    mask_center: bool = False
    dilation: int = 1

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask_center: bool = False,
        dilation: int = 1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mask_center = mask_center
        self.dilation = dilation

        # Create the mask using jnp instead of np
        mask = jnp.ones((1, self.kernel_size), dtype=jnp.float32)
        mask = mask.at[0, self.kernel_size // 2 + 1 :].set(0)
        # For the first convolution, we will also mask the center pixel
        if self.mask_center:
            mask = mask.at[0, self.kernel_size // 2].set(0)

        # Initialize the MaskedConv module
        self.conv = MaskedConv(
            key=key,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            mask=mask,
            dilation=self.dilation,
        )

    def __call__(
        self, x: Float[Array, "in_channel height width"]
    ) -> Float[Array, "out_channel height width"]:
        return self.conv(x)


class GatedMaskedConv(eqx.Module):
    in_channels: int

    conv_vertical: VerticalStackConv
    conv_horizontal: HorizontalStackConv
    conv_horizontal_1x1: eqx.nn.Conv2d
    conv_vertical_to_horizontal: eqx.nn.Conv2d

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        dilation: int = 1,
    ):

        self.in_channels = in_channels

        vertical_key, horizontal_key, vertical_to_horizontal_key, horizontal_1x1_key = (
            jax.random.split(key, 4)
        )

        # Double the number of channels in the output so that later we can split them
        # into the value and gate
        self.conv_vertical = VerticalStackConv(
            key=vertical_key,
            in_channels=in_channels,
            out_channels=2 * in_channels,
            kernel_size=3,
            mask_center=False,
            dilation=dilation,
        )

        self.conv_horizontal = HorizontalStackConv(
            key=horizontal_key,
            in_channels=in_channels,
            out_channels=2 * in_channels,
            kernel_size=3,
            mask_center=False,
            dilation=dilation,
        )

        self.conv_vertical_to_horizontal = eqx.nn.Conv2d(
            key=vertical_to_horizontal_key,
            in_channels=2 * in_channels,
            out_channels=2 * in_channels,
            kernel_size=(1, 1),
        )

        self.conv_horizontal_1x1 = eqx.nn.Conv2d(
            key=horizontal_1x1_key,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1),
        )

    def __call__(self, v_stack: jax.Array, h_stack: jax.Array):
        v_val, v_gate = jnp.split(self.conv_vertical(v_stack), 2, axis=0)
        v_stack_out = jax.nn.tanh(v_val) * jax.nn.sigmoid(v_gate)

        h_stack_features = self.conv_horizontal(h_stack)
        # Instead of directly passing passing vertical features into horizontal stack,
        # the 1x1 convolution(self.conv_vertical_to_horizontal) allows for a learnable
        # tranformation, also gives compatible dims to be added to h_stack_features
        h_stack_features += self.conv_vertical_to_horizontal(
            self.conv_vertical(v_stack)
        )
        # Split along the channel axis in the array of (channel, height, width)
        h_val, h_gate = jnp.split(h_stack_features, 2, axis=0)

        # Gated activation
        h_stack_features = jax.nn.tanh(h_val) + jax.nn.sigmoid(h_gate)
        # Apply 1x1 convolution to reduce the dimensionality from 2 * in_channels to in_channels
        # This also allows the h_stack_out to align with the residual connection (h_stack)
        h_stack_out = self.conv_horizontal_1x1(h_stack_features)
        h_stack_out += h_stack  # Residual connection

        return v_stack_out, h_stack_out


class PixelCNN(eqx.Module):
    in_channels: int
    hidden_count: int

    vstack_conv: VerticalStackConv
    hstack_conv: HorizontalStackConv
    conv_layers: List[GatedMaskedConv]
    out_conv: eqx.nn.Conv2d

    def __init__(
        self, key: jax.Array, in_channels: int, hidden_count: int
    ):  # TODO change the key type hint to a jaxtyping hint
        vstack_key, hstack_key, gated_key, out_key = jax.random.split(key, 4)
        self.in_channels, self.hidden_count = in_channels, hidden_count

        self.vstack_conv = VerticalStackConv(
            key=vstack_key,
            in_channels=in_channels,
            out_channels=hidden_count,
            kernel_size=3,
            mask_center=True,
            dilation=1,
        )

        self.hstack_conv = HorizontalStackConv(
            key=hstack_key,
            in_channels=in_channels,
            out_channels=hidden_count,
            kernel_size=3,
            mask_center=True,
            dilation=1,
        )

        g_key1, g_key2, g_key3, g_key4, g_key5, g_key6, g_key7 = jax.random.split(
            gated_key, 7
        )

        self.conv_layers = [
            GatedMaskedConv(g_key1, hidden_count),
            GatedMaskedConv(g_key2, hidden_count, dilation=2),
            GatedMaskedConv(g_key3, hidden_count),
            GatedMaskedConv(g_key4, hidden_count, dilation=4),
            GatedMaskedConv(g_key5, hidden_count),
            GatedMaskedConv(g_key6, hidden_count, dilation=2),
            GatedMaskedConv(g_key7, hidden_count),
        ]

        # Convert the learnt features from the previous layers into a probability distribution over possible pixel values.
        # This is done by applying a 1x1 convolution to the hidden features. This combines the features from all channels.
        # For each input channel, we output 256 possible values (0-255).
        self.out_conv = eqx.nn.Conv2d(
            key=out_key,
            in_channels=hidden_count,
            out_channels=in_channels * 256,
            kernel_size=(1, 1),
        )

    def get_logits(
        self, x: Float[Array, "channel height width"]
    ) -> Float[Array, "256 channel height width"]:
        # scale input from 0-255 to -1 to 1
        x = (x.astype(jnp.float32) * 255.0) * 2.0 - 1.0

        v_stack = self.vstack_conv(x)
        h_stack = self.hstack_conv(x)

        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)

        # elu for smooth gradients of negative inputs
        out = self.out_conv(jax.nn.elu(h_stack))
        return out

    def __call__(self, x: Float[Array, "channel height width"]) -> Scalar:
        logits = self.get_logits(x)
        labels = x.astype(jnp.int32)

        # compute negative log likelihood
        logits = einops.rearrange(
            logits,
            "(in_channels n) w h -> n in_channels w h",
            in_channels=self.in_channels, n=256,
        )

        import pdb; pdb.set_trace()
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        # we need to transform the densities returned by model (in logit space) back to image space [0-256]
        # and compute bits per dims
        bpd = nll.mean() * jnp.log2(jnp.exp(1))
        return bpd

    def sample(
        self,
        key: PRNGKeyArray,
        image_shape: Tuple[int, int, int],
        image: Optional[Float[Array, "channel height width"]] = None,
    ):
        image = image if image else jnp.zeros(image_shape, dtype=jnp.int32) - 1
        get_logits = jax.jit(lambda image: self.get_logits(image))

        def _sample(key, image, c, h, w):
            logits = get_logits(image)
            # filter out and get the logits only for the
            # c, h, w we want to sample for.
            logits = logits[:, c, h, w]
            sampled = jax.random.categorical(key, logits, axis=0)
            return sampled

        channel, height, width = image_shape
        for c in range(channel):
            for h in range(height):
                for w in range(width):
                    key, sampling_key = jax.random.split(key, 2)
                    sampled = _sample(sampling_key, image, c, h, w)
                    image = image.at[:, c, h, w].set(sampled)
        return image
