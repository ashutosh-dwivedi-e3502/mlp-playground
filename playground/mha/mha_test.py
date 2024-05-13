import unittest
import jax

from jax import random

# from playground.mha import model
from . import model


class TestMHA(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.main_rng = random.PRNGKey(42)

    def test_scaled_dot_product(self):
        seq_len, d_k = 3, 2
        _, rand1 = random.split(self.main_rng)
        qkv = random.normal(rand1, (3, seq_len, d_k))
        q, k, v = qkv[0], qkv[1], qkv[2]
        values, attention = model.scaled_dot_product(q, k, v)
        assert values.shape == (3, 2), f"{values.shape=}"
        assert attention.shape == (3, 3), f"{attention.shape=}"

    def test_mha(self):
        main_rng, x_rng = random.split(self.main_rng)
        # Create attention
        mh_attn = model.MultiHeadAttention(embed_dim=128, num_heads=4, key=main_rng)
        # create random input
        x = random.normal(x_rng, (16, 128))
        output, attention = mh_attn(x)

        assert output.shape == (16, 128), f"{output.shape}"
        assert attention.shape == (4, 16, 16), f"{attention.shape}"

    def test_encoder_block(self):
        ## Test EncoderBlock implementation
        # Example features as input
        main_rng, x_rng = random.split(self.main_rng)
        x = random.normal(x_rng, (16, 128))
        # Create encoder block
        encblock = model.EncoderBlock(
            input_dim=128,
            num_heads=4,
            dim_feedforward=512,
            dropout_prob=0.1,
            key=self.main_rng,
        )
        # Initialize parameters of encoder block with random key and inputs
        main_rng, init_rng, dropout_rng = random.split(main_rng, 3)
        # Apply encoder block with parameters on the inputs
        out = encblock(x, dropout_rng)
        assert out.shape == (16, 128)

    def test_transformer_encoder(self):
        main_rng, x_rng = random.split(self.main_rng)
        x = random.normal(x_rng, (16, 128))

        # Create Transformer encoder
        main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
        transfomer_encoder = model.TransformerEncoder(
            num_layers=5,
            input_dim=128,
            num_heads=4,
            dim_feedforward=256,
            dropout_prob=0.15,
            key=init_rng,
        )
        # Initialize parameters of transformer with random key and inputs
        out = transfomer_encoder(x, dropout_init_rng)
        out_inference = transfomer_encoder(x, dropout_init_rng, train=False)


if __name__ == "__main__":
    unittest.main()
