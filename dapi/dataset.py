import json
import os
from shutil import copy
import itertools

from dapi.utils import open_image


def parse_predictions(prediction_dir,
                      real_class,
                      fake_class):
    '''Parse cycle-GAN predictions from prediction dir.

    Args:

        prediction_dir: (''str'')

            Path to cycle-GAN prediction dir

        real_class: (''int'')

            Real class output index

        fake_class: (''int'')

            Fake class output index
    '''

    files = [os.path.join(prediction_dir, f) for f in os.listdir(prediction_dir)]
    real_imgs = [f for f in files if f.endswith("real.png")]
    fake_imgs = [f for f in files if f.endswith("fake.png")]
    pred_files = [f for f in files if f.endswith("aux.json")]

    # try to get an integer id:
    img_ids = []
    for id_idx in range(len(real_imgs[0].split("/")[-1].split("_"))):
        try:
            img_ids = [int(f.split("/")[-1].split("_")[id_idx]) for f in real_imgs]
            transform = lambda x: int(x)
            no_numbers = False
            break
        except ValueError:
            pass

    #If that does not work use string and rename:
    if not img_ids:
        img_ids = [f.split("/")[-1].split("_")[0] for f in real_imgs]
        id_idx = 0
        transform = lambda x: x
        no_numbers = True


    ids_to_data = {}
    for img_id in img_ids:
        real = [f for f in real_imgs if img_id == transform(f.split("/")[-1].split("_")[id_idx])]
        fake = [f for f in fake_imgs if img_id == transform(f.split("/")[-1].split("_")[id_idx])]
        aux = [f for f in pred_files if img_id == transform(f.split("/")[-1].split("_")[id_idx])]
        assert(len(real) == 1)
        assert(len(fake) == 1)
        assert(len(aux) == 1)

        real = real[0]
        fake = fake[0]
        aux = aux[0]
        aux_data = json.load(open(aux, "r"))
        aux_real = aux_data["aux_real"][real_class]
        aux_fake = aux_data["aux_fake"][fake_class]

        ids_to_data[img_id] = (real, fake, aux_real, aux_fake)

    return ids_to_data

def create_filtered_dataset(ids_to_data, data_dir, threshold=0.8):
    '''Filter out failed translations (f(x)<threshold)

    Args:

        ids_to_data: (''Dict'')

            Dictionary of image ids to image data as returned by
            parse_predictions

        data_dir: (''str'')

            Output path for filtered datasets
    '''

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    idx = 0
    for img_id, data in ids_to_data.items():
        if (data[2] > threshold) and (data[3] > threshold):
            copy(data[0], os.path.join(data_dir + f"/real_{idx}.png"))
            copy(data[1], os.path.join(data_dir + f"/fake_{idx}.png"))
            idx += 1
