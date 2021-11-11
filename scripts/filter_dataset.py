import argparse

from dac.dataset import parse_predictions, create_filtered_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir")
parser.add_argument("--out_dir")
parser.add_argument("--real_class_idx", type=int)
parser.add_argument("--fake_class_idx", type=int)
parser.add_argument("--threshold", required=False, default=0.8, type=float)


def filter_predictions(in_dir, out_dir, real_class_idx, fake_class_idx, threshold):
    ids_to_data = parse_predictions(in_dir, real_class_idx, fake_class_idx)
    create_filtered_dataset(ids_to_data, out_dir, threshold)


def filter_all():
    datasets = ["apples_oranges", "horses_zebras", "summer_winter"]
    ids = [(0,1), (1,0)]
    nets = ["vgg", "res"]

    for dataset in datasets:
        for class_ids in ids:
            for net in nets:
                real_class_idx = class_ids[0]
                fake_class_idx = class_ids[1]
                class_from = dataset.split("_")[real_class_idx]
                class_to = dataset.split("_")[fake_class_idx]

                if net == "vgg":
                    net_in = "vgg2d"
                else:
                    net_in = net

                in_dir = f"/groups/funke/home/ecksteinn/dapi/ecksteinn/data/{dataset}/train/cycle_gan/{dataset}/results/{net_in}_{class_from}/{dataset}/test_latest/images"
                out_dir = f"/groups/funke/home/ecksteinn/dapi/ecksteinn/data/translated/{dataset}/{net}/{class_from}_{class_to}"

                filter_predictions(in_dir, out_dir, real_class_idx, fake_class_idx, 0.8)



if __name__ == "__main__":
    filter_all()
    """
    args = parser.parse_args()
    filter_predictions(args.in_dir,
                       args.out_dir,
                       args.real_class_idx,
                       args.fake_class_idx,
                       args.threshold)
    """

