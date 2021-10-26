from PIL import Image
import os
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--img_dirs", type=str, nargs="+", required=True)
parser.add_argument("--extension", type=str, default=".jpg", required=False)

def get_avg_size(img_dirs, extension=".jpg"):
    """Get average width, height of images in 
       given dirs. Useful for specifying load sizes
       for cycle GAN training.
    """
    img_paths = []
    for d in img_dirs:
        img_paths_in_d = [os.path.join(d, f) for f in os.listdir(d) if
                          f.endswith(f"{extension}")]
        img_paths.extend(img_paths_in_d)

    width = []
    height = []
    for img_path in tqdm(img_paths):
        im = Image.open(img_path)
        w, h = im.size
        width.append(w)
        height.append(h)

    return np.mean(width), np.mean(height)

if __name__ == "__main__":
    args = parser.parse_args()
    avg_width, avg_height = get_avg_size(args.img_dirs, args.extension)
    print("avg width: ", avg_width, "avg_height: ", avg_height, "avg_side: ",
          (avg_width + avg_height)/2)

