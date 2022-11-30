import os
from PIL import Image
import torch
from fastai.vision.all import get_image_files
from pathlib2 import Path
import configparser
import pandas as pd
import numpy as np


def flatten_image(pil_image):
    """
    pil_image: image as returned from PIL Image
    """
    if len(pil_image.shape) == 2:
        return np.expand_dims(pil_image, axis=0)
    return np.expand_dims(np.array(pil_image[:, :, 0]), axis=0)


def normalize_image(image):
    """
    image: 2D input image
    """
    return (image.astype(np.float32) / 255.0 - 0.5) / 0.5


def open_image(image_path, flatten=False, normalize=True):
    im = np.asarray(Image.open(image_path))
    if flatten:
        im = flatten_image(im)
    else:
        if len(im.shape) == 2:
            # sanity check if 3 channel image is 1 d we convert to 3 channel
            im = np.asarray(Image.open(image_path).convert("RGB"))
        im = im.T
    if normalize:
        im = normalize_image(im)
    return im


class GetDataset:
    """
    generic dataset class for creating pair images
    """

    def __init__(
        self,
        class_names: str,
        model_type: str,
        flatten: bool,
        txt_fln="reals.txt",
        folder="dapi_data",
        config_name="dapi_data/experiments/configs/synapses.ini",
        raw_files_folder="dapi_data/data/raw/synapses/",
    ):
        self.flatten = flatten
        df = pd.read_csv(txt_fln, header=None)
        df.columns = ["path"]
        df = df[df["path"].str.contains(f"vgg/{class_names}", case=False)].reset_index(
            drop=True
        )
        df["path"] = df["path"].apply(lambda x: folder + "/" + x)

        self.fns_from = df["path"].to_list()

        self.from_class, self.to_class = class_names.split("_")

        fns_to = sorted(get_image_files(Path(raw_files_folder) / self.to_class))
        self.fns_to = fns_to[: len(self.fns_from)]
        assert len(self.fns_from) == len(self.fns_to)

        config = configparser.ConfigParser()
        config.read(config_name)
        self.class_name_dict = {
            k: i for i, k in enumerate(config[model_type]["class_names"].split(","))
        }

    def __len__(self):
        return len(self.fns_from)

    def __getitem__(self, idx):
        query_image = open_image(self.fns_from[idx], flatten=self.flatten)
        query_image_name = self.fns_from[idx]
        query_label = self.class_name_dict[self.from_class]
        distractor_image = open_image(self.fns_to[idx], flatten=self.flatten)
        distractor_label = self.class_name_dict[self.to_class]
        distractor_image_name = self.fns_to[idx]
        return (
            query_image,
            query_label,
            distractor_image,
            distractor_label,
            Path(query_image_name),
            distractor_image_name,
        )

    def get_num_clases(self):
        return len(self.class_name_dict)
