import os
import itertools
import shutil
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Path to dataset", required=True)
parser.add_argument("--classes", nargs="+", help="Network Classes", required=True)

def prepare_dataset(data_dir, classes, train=1, val=0, test=0):
    class_combinations = list(itertools.combinations(classes, 2))

    out_dir = os.path.join(data_dir, "cycle_gan")
    assert(train + val + test == 1)

    splits = {"train": [0,train],
              "val": [train, train+val],
              "test": [train+val, train+val+test]}

    class_to_files = {c: None for c in classes}
    for c in classes:
        class_dir = os.path.join(data_dir, str(c))

        f_in_d = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if
                f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".jpg")]
        class_to_files[c] = f_in_d

    for pair in class_combinations:
        f_0 = class_to_files[pair[0]]
        f_1 = class_to_files[pair[1]]

        class_pairs = {"A": f_0,
                       "B": f_1}

        pair_dir = os.path.join(out_dir, f"{pair[0]}_{pair[1]}")

        for split, split_fraction in splits.items():
            for direction, files in class_pairs.items():
                split_dir = os.path.join(pair_dir, split + direction)
                if not os.path.exists(split_dir):
                    os.makedirs(split_dir)

                files_in_split = files[split_fraction[0]*len(files):split_fraction[1]*len(files)]
                for f in tqdm(files_in_split):
                    shutil.copy(f, os.path.join(split_dir, f.split("/")[-1]))

if __name__ == "__main__":
    args = parser.parse_args()
    prepare_dataset(args.data_dir, [i for i in args.classes])
