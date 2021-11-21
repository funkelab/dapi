import os
import numpy as np
import sys
from tqdm import tqdm
import argparse
import json

from dapi.utils import open_image, save_image, get_image_pairs
from dapi.mask import get_mask
from dapi.attribute import get_attribution

parser = argparse.ArgumentParser()
parser.add_argument('--worker', type=int, required=True)
parser.add_argument('--id_min', type=int, required=True)
parser.add_argument('--id_max', type=int, required=True)
parser.add_argument('--img_dir', type=str, required=True)
parser.add_argument('--real_classes', nargs="+", required=True)
parser.add_argument('--fake_classes', nargs="+", required=True)
parser.add_argument('--class_names', nargs="+", required=True)
parser.add_argument('--net_module', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--input_shape', nargs="+", type=int, required=True)
parser.add_argument('--output_classes', type=int, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--abs_attr', action="store_true")
parser.add_argument('--methods', nargs="+", default=["ig", "grads", "dl", "gc", "ggc", "ingrad", "random", "residual"])
parser.add_argument('--downsample_factors', nargs="+")
parser.add_argument('--bidirectional', type=int, required=True)
parser.add_argument('--max_images', type=int, required=False, default=None)
parser.add_argument('--channels', type=int, required=True)

def run_worker(worker,id_min,id_max,img_dir, real_classes,
               fake_classes, net_module, checkpoint,
               input_shape, output_classes, out_dir,
               abs_attr, methods, class_names, downsample_factors,
               bidirectional, max_images, channels, write_opt=True):

    bidirectional = bool(bidirectional)
    reals = []
    fakes = []
    for real_class, fake_class in zip(real_classes, fake_classes):
        image_pairs = get_image_pairs(img_dir, real_class, fake_class)
        reals_dir = [(p[0],class_names.index(str(real_class))) for p in image_pairs]
        fakes_dir = [(p[1],class_names.index(str(fake_class))) for p in image_pairs]

        if max_images is not None:
            reals.extend(reals_dir[:max_images])
            fakes.extend(fakes_dir[:max_images])
        else:
            reals.extend(reals_dir)
            fakes.extend(fakes_dir)


    for i in tqdm(range(id_min, id_max), position=worker):
        real = reals[i]
        fake = fakes[i]
        with HiddenPrints():
            img_idx = i
            img_dir = os.path.join(out_dir, f"{i}")

            real_img = real[0]
            fake_img = fake[0]
            real_class = real[1]
            fake_class = fake[1]

            real_img = open_image(real_img, flatten=channels==1, normalize=False)
            fake_img = open_image(fake_img, flatten=channels==1, normalize=False)

            if methods is None:
                attrs, attrs_names = get_attribution(real_img, fake_img, real_class,
                                                     fake_class, net_module, checkpoint,
                                                     input_shape, channels,
                                                     output_classes=output_classes,
                                                     bidirectional=bidirectional,
                                                     downsample_factors=downsample_factors)
            else:
                attrs, attrs_names = get_attribution(real_img, fake_img, real_class,
                                                     fake_class, net_module, checkpoint,
                                                     input_shape, channels, methods,
                                                     output_classes=output_classes,
                                                     bidirectional=bidirectional,
                                                     downsample_factors=downsample_factors)


            for attr, name in zip(attrs, attrs_names):
                if abs_attr:
                    attr = np.abs(attr)

                result_dict, img_names, imgs_all = get_mask(attr, real_img, fake_img,
                                                            real_class, fake_class,
                                                            net_module, checkpoint, input_shape,
                                                            channels, output_classes=output_classes,
                                                            downsample_factors=downsample_factors)




                method_dir = os.path.join(img_dir, name)
                if not os.path.exists(method_dir):
                    os.makedirs(method_dir)

                    with open(os.path.join(method_dir, "results.txt"), 'w+') as f:
                        print(result_dict, file=f)

                if write_opt:
                    thr_idx, thr, mask_size, mask_score = get_optimal_mask(result_dict, input_shape[0])
                    imgs_opt = imgs_all[thr_idx]
                    imgs_dir = os.path.join(method_dir, "opt_images")
                    if not os.path.exists(imgs_dir):
                        os.makedirs(imgs_dir)

                    for img_opt, img_name in zip(imgs_opt, img_names):
                        out_path = os.path.join(imgs_dir, f"{img_name}.png")
                        save_image(img_opt, out_path)

                    with open(f'{imgs_dir}/img_info.json', "w+") as f:
                        json.dump({"real_class": real_class,
                                   "fake_class": fake_class,
                                   "thr": thr,
                                   "mask_size": mask_size,
                                   "mask_score": mask_score,
                                   "thr_idx": int(thr_idx)}, f)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def parse_args(args):
    arg_dict = vars(args)
    down = arg_dict["downsample_factors"]
    if down[0] != "None":
        down = [int(k) for k in arg_dict["downsample_factors"]]
        down = [(down[i], down[i+1]) for i in range(0,len(down),2)]
    else:
        down = None
    arg_dict["downsample_factors"] = down
    return arg_dict

def get_optimal_mask(result_dict, size):
    def ascore(m_s, m_n):
        return m_n**2 + (1 - m_s)**2

    ascores = []
    thrs = []
    mask_sizes = []
    mask_scores = []
    for thr, m in result_dict.items():
        mask_score = m[0]
        mask_size = m[1]/float(size)**2
        ascores.append(ascore(mask_score, mask_size))
        thrs.append(thr)
        mask_sizes.append(mask_size)
        mask_scores.append(mask_score)

    thr_idx = np.argmin(ascores)
    thr = thrs[thr_idx]
    mask_size = mask_sizes[thr_idx]
    mask_score = mask_scores[thr_idx]

    return thr_idx, thr, mask_size, mask_score

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dict = parse_args(args)
    run_worker(**arg_dict)
