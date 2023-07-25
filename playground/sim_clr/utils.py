import os
import urllib.request
from urllib.error import HTTPError

from .constants import CHECKPOINT_PATH


def download_pre_trained():
    """download pre-trained"""
    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial17/"
    # Files to download
    pretrained_files = ["SimCLR.ckpt", "ResNet.ckpt",
                        "tensorboards/SimCLR/events.out.tfevents.SimCLR",
                        "tensorboards/classification/ResNet/events.out.tfevents.ResNet"]
    pretrained_files += [f"LogisticRegression_{size}.ckpt" for size in [10, 20, 50, 100, 200, 500]]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if "/" in file_name:
            os.makedirs(file_path.rsplit("/",1)[0], exist_ok=True)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as exception:
                print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", exception)
