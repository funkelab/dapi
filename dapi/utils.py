from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def flatten_image(pil_image):
    """
    pil_image: image as returned from PIL Image
    """
    return np.expand_dims(
        np.array(
            pil_image[:, :, 0],
            dtype=np.float32),
        axis=0)


def normalize_image(image):
    """
    image: 2D input image
    """
    return (image.astype(np.float32)/255. - 0.5)/0.5


def open_image(image_path, flatten=True, normalize=True):
    im = np.asarray(Image.open(image_path))
    if flatten:
        im = flatten_image(im)
    else:
        im = im.T
    if normalize:
        im = normalize_image(im)
    return im


def image_to_tensor(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = torch.tensor(image, device=device)
    if len(np.shape(image)) == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif len(np.shape(image)) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    else:
        raise ValueError("Input shape not understood")

    return image_tensor


def save_image(array, image_path, renorm=True, norm=False):
    if renorm:
        array = (array * 0.5 + 0.5) * 255
    if norm:
        array /= np.max(np.abs(array))
        array *= 255

    if np.shape(array)[0] == 1:
        # greyscale
        array = np.concatenate([array.astype(np.uint8),]*3, axis=0).T
        plt.imsave(image_path, array, cmap='gray')
    else:
        plt.imsave(image_path, array.T.astype(np.uint8))
