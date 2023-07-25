import dm_pix
import jax
import numpy as np


from jax import numpy as jnp
from jax import random
from torchvision import transforms
from torchvision.datasets import STL10

class ContrastiveTransformations(object):
    """Transformations for simclr"""

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = img / 255.
    return img

def augment_image(rng, img):
    rngs = random.split(rng, 8)
    # Random left-right flip
    img = dm_pix.random_flip_left_right(rngs[0], img)
    # Color jitter
    img_jt = img
    img_jt = img_jt * random.uniform(rngs[1], shape=(1,), minval=0.5, maxval=1.5)  # Brightness
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    img_jt = dm_pix.random_contrast(rngs[2], img_jt, lower=0.5, upper=1.5)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    img_jt = dm_pix.random_saturation(rngs[3], img_jt, lower=0.5, upper=1.5)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    img_jt = dm_pix.random_hue(rngs[4], img_jt, max_delta=0.1)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    should_jt = random.bernoulli(rngs[5], p=0.8)
    img = jnp.where(should_jt, img_jt, img)
    # Random grayscale
    should_gs = random.bernoulli(rngs[6], p=0.2)
    img = jax.lax.cond(should_gs,  # Only apply grayscale if true
                       lambda x: dm_pix.rgb_to_grayscale(x, keep_dims=True),
                       lambda x: x,
                       img)
    # Gaussian blur
    sigma = random.uniform(rngs[7], shape=(1,), minval=0.1, maxval=2.0)
    img = dm_pix.gaussian_blur(img, sigma=sigma[0], kernel_size=9)
    # Normalization
    img = img * 2.0 - 1.0
    return img


def get_stl_dataset(dataset_path, size=96):
    contrast_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          image_to_numpy])
    unlabeled_data = STL10(root=dataset_path, split='unlabeled', download=True,
                       transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    train_data_contrast = STL10(root=dataset_path, split='train', download=True,
                            transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    return unlabeled_data, train_data_contrast