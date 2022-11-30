from einops import rearrange
import numpy as np
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from skimage.io import imsave
from pdb import set_trace
from models import Vgg2D, ResNet


def permute(image, p):
    # shuffles image according to permutaton matrix
    fw = np.sqrt((p.shape[-1])).astype("int")
    patch_size = image.shape[-1] // fw
    x = (
        rearrange(
            image,
            "b c (h p1) (w p2) -> b (p1 p2 c) (h w)",
            p1=patch_size,
            p2=patch_size,
        )
        @ p
    )
    x = rearrange(
        x,
        "b (p1 p2 c) (h w)-> b c (h p1) (w p2)",
        p1=patch_size,
        p2=patch_size,
        h=fw,
        w=fw,
    )
    return x


def mix(image_1, image_2, a):
    # mix two images according to a
    fw = np.sqrt((a.shape[-1])).astype("int")
    a = F.interpolate(a.reshape(1, 1, fw, fw), (image_1.shape)[2:])
    x = torch.lerp(image_1, image_2, a)
    return x


def create_hybrid_sequence(query_image, distractor_image, edits, num_patches):

    # query_image: (1, h, w)
    # distractor_image: (1, h, w)

    x_hs = []

    for copy_from, copy_to in tqdm(edits):

        x_h = copy_patch(
            source=distractor_image,
            target=query_image,
            copy_from=copy_from,
            copy_to=copy_to,
            num_patches=num_patches,
        )
        x_hs.append(x_h)

        query_image = x_h

    return x_hs


def copy_patch(source, target, copy_from, copy_to, num_patches):

    width = source.shape[-1]
    height = source.shape[-2]
    # we assume that number of patches is same in x and y
    num_patches_x = int(np.sqrt(num_patches))
    num_patches_y = int(np.sqrt(num_patches))
    patch_width = width // num_patches_x
    patch_height = height // num_patches_y
    # flatten images
    # source: (c, h, w)
    # target: (c, h, w)
    source = rearrange(
        source, "c (h p1) (w p2) -> (p1 p2 c) (h w)", p1=patch_height, p2=patch_width
    )
    target = rearrange(
        target.clone(),
        "c (h p1) (w p2) -> (p1 p2 c) (h w)",
        p1=patch_height,
        p2=patch_width,
    )
    # source: (c*pw*ph, n)    n = num_patches
    # target: (c*pw*ph, n)    n = num_patches

    target[:, copy_to] = source[:, copy_from]

    # revert flattening
    # target: (c*pw*ph, n)    n = num_patches
    target = rearrange(
        target,
        "(p1 p2 c) (h w) -> c (h p1) (w p2)",
        p1=patch_height,
        p2=patch_width,
        h=num_patches_y,
        w=num_patches_x,
    )
    # target: (c, h, w)

    return target


def scale(x):
    return (((x * 0.5) + 0.5) * 255.0).astype("uint8")


def save_q_and_d(name, q, d):
    os.makedirs(name, exist_ok=True)
    # TODO: support RGB images
    imsave(name + "/query_image.png", scale(q.transpose(1, 2, 0)))
    imsave(name + "/distractor_image.png", scale(d.transpose(1, 2, 0)))


def save_image_sequnce(name, hybrid_sequence):
    os.makedirs(name, exist_ok=True)
    hybrid_sequence = hybrid_sequence
    n = len(hybrid_sequence)
    for i in range(n):
        # TODO: support RGB images
        img = hybrid_sequence[i].numpy()
        imsave(name + ("/seq_%.4d.png" % i), scale(img.transpose(1, 2, 0)))


def extract_features(x, model):
    x = x.cuda()
    # x: (c, h, w)
    with torch.no_grad():
        x = rearrange(x, "c h w -> () c h w")
        # x: (1, c, h, w)
        out = model.features(x)
        # out: (1, c, h', w')  h' < h, w' < w
    out = rearrange(out[0], "l h w -> l (h w)")
    # out: (1, c, n)  n = h'*w'
    return out


def load_pretrained_model_weights(
    model_name,
    chk_name,
    downsample_factors,
    input_size=(256, 256),
    input_channels=3,
    output_classes=2,
):
    model = eval(model_name)(
        input_size=input_size,
        input_channels=input_channels,
        output_classes=output_classes,
        downsample_factors=downsample_factors,
    )
    try:
        md_dict = torch.load(chk_name)["model_state_dict"]
    except:
        md_dict = torch.load(chk_name)
    model.load_state_dict(state_dict=md_dict)
    model.eval()
    return model