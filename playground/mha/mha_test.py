import unittest

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
        print("Q\n", q)
        print("K\n", k)
        print("V\n", v)
        print("Values\n", values)
        print("Attention\n", attention)
        assert values.shape == (3, 2), f"{values.shape=}"
        assert attention.shape == (3, 3), f"{attention.shape=}"
    
    
    def test_mha(self):
        main_rng, x_rng = random.split(self.main_rng)
        # Create attention
        mh_attn = model.MultiHeadAttention(embed_dim=128, num_heads=4, key=main_rng)
        # create random input 
        x = random.normal(x_rng, (3, 16, 128))
        output, attention = mh_attn(x)

        assert output.shape == (3, 16, 128), f"{output.shape}"
        assert attention.shape == (3, 4, 16, 16), f"{attention.shape}"


if __name__ == '__main__':
    unittest.main()