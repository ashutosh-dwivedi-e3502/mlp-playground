import os

import settings

BASE_DIR = os.path.pardir
DATASET_PATH = os.path.join(settings.BASE_DIR, "data")
CHECKPOINT_PATH = os.path.join(settings.BASE_DIR, "saved_models/tutorial17_jax")
