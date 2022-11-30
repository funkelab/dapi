from dapi.activations import get_activation_dict, get_layer_activations, \
    project_layer_activations_to_input_rescale
from dapi.gradients import get_gradients_from_layer
from dapi.utils import normalize_image
import collections
import numpy as np
import torch


def get_sgc(
        real_img,
        fake_img,
        real_class,
        fake_class,
        classifier,
        layer_name=None):
    """
        real_img: Unnormalized (0-255) 2D image

        fake_img: Unnormalized (0-255) 2D image

        *_class: Index of real and fake class corresponding to network output

        TODO: add classifier

        layer_name: Name of the conv layer to use (defaults to last)

    Args:

        real_img: (array-like)

            Real image to run attribution on.

        fake_img: (array-like)

            Counterfactual image typically created by a cycle GAN.

        real_class: (int)

            Class index of real image. Must correspond to networks output
            class.

        fake_class: (''int'')

            Class index of fake image. Must correspond to networks output
            class.

        classifier: (torch module)

            The classifier network to use.

        layer_name: (string)

            Name of the conv layer to use (defaults to last).
    """

    # get input shape and number of channels
    channels, height, width = real_img.shape
    input_shape = (height, width)

    imgs = [normalize_image(real_img), normalize_image(fake_img)]
    classes = [real_class, fake_class]

    if layer_name is None:
        last_conv_layer = [
            (name, module)
            for name, module in classifier.named_modules()
            if type(module) == torch.nn.Conv2d
        ][-1]
        layer_name = last_conv_layer[0]

    grads = []
    for x, y in zip(imgs, classes):
        grads.append(get_gradients_from_layer(classifier, x, y, layer_name))

    print(f"GRAD SHAPE {grads[0].shape}")

    acts_real = collections.defaultdict(list)
    acts_fake = collections.defaultdict(list)

    acts_real, _ = get_activation_dict(
        classifier,
        [imgs[0]],
        acts_real)
    acts_fake, _ = get_activation_dict(
        classifier,
        [imgs[1]],
        acts_fake)

    acts = [acts_real, acts_fake]

    layer_acts = []
    for act in acts:
        layer_acts.append(get_layer_activations(act, layer_name))

    delta_fake = grads[1] * (layer_acts[0] - layer_acts[1])
    delta_real = grads[0] * (layer_acts[1] - layer_acts[0])

    delta_fake_projected = project_layer_activations_to_input_rescale(
        delta_fake,
        (input_shape[0], input_shape[1]))[0, :, :, :]
    delta_real_projected = project_layer_activations_to_input_rescale(
        delta_real,
        (input_shape[0], input_shape[1]))[0, :, :, :]

    channels = np.shape(delta_fake_projected)[0]
    gc_0 = np.zeros(np.shape(delta_fake_projected)[1:])
    gc_1 = np.zeros(np.shape(delta_real_projected)[1:])

    for c in range(channels):
        gc_0 += delta_fake_projected[c, :, :]
        gc_1 += delta_real_projected[c, :, :]

    gc_0 = np.abs(gc_0)
    gc_1 = np.abs(gc_1)
    gc_0 /= np.max(np.abs(gc_0))
    gc_1 /= np.max(np.abs(gc_1))
    gc_0 = np.stack([gc_0,]*channels, axis=0)
    gc_1 = np.stack([gc_1,]*channels, axis=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.tensor(gc_0, device=device), torch.tensor(gc_1, device=device)
