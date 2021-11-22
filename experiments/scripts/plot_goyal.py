import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Plot goyal et al. comparison')
parser.add_argument('--dset', help="name of dataset, e.g. summer_winter")
parser.add_argument('--goyal_dset_path', help="Path to goyal et al. results")
parser.add_argument('--dapi_dset_path', help="Path to dapi results")
parser.add_argument('--net', help="Network (res or vgg)")
parser.add_argument('--dapi_method', help="Attribution method to plot, one of ig, dl, gc, ggc, ingrad, residual")


mpl.rcParams['figure.dpi'] = 300
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)
mpl.rc('text', usetex=False)
mpl.rc('axes', linewidth=2)
mpl.rc('font', weight='bold')

def parse_dapi_results(res_dir):
    methods_to_results = {}
    img_dirs = [os.path.join(res_dir, d) for d in os.listdir(res_dir) if d[0] != "."]
    for img_dir in img_dirs:
        methods = os.listdir(img_dir)
        methods_dirs = [os.path.join(img_dir, d) for d in methods if d[0] != "."]
        for method_dir, method in zip(methods_dirs, methods):
            with open(os.path.join(method_dir, "results.txt"), "r") as f:
                content = f.read()
                try:
                    method_dict = eval(content)
                except NameError:
                    # NAN
                    continue

                if method in methods_to_results:
                    k = 0
                    for thr in method_dict:
                        methods_to_results[method][k]["mrf"].append(method_dict[thr][0])
                        methods_to_results[method][k]["mask_size"].append(method_dict[thr][1])
                        k += 1

                else:
                    methods_to_results[method] = {k: {"mrf": [method_dict[list(method_dict)[k]][0]],
                                                      "mask_size": [method_dict[list(method_dict)[k]][1]]} for k in range(len(method_dict))}

    return sort_results(methods_to_results)


def plot_dapi(base_to_res, experiment, net, leg=True, methods=None, bl=False):
    if experiment == "mnist":
        size = 28
    elif experiment in ["horses_zebras", "apples_oranges", "summer_winter"]:
        size = 256
    else:
        size = 128

    if leg:
        leg_str = "_leg"
    else:
        leg_str = ""
    out_path = f"dapi_plot_{experiment}_{net}{leg_str}.png"

    method_to_color = {"dl": "springgreen",
                       "gc": "deeppink",
                       "ggc": "blue",
                       "ig": "orange",
                       "ingrad": "purple",
                       "baseline": "dimgray"}

    plot_lines_dapi = []
    plot_lines_base = []

    if methods is None:
        methods = ["dl", "gc", "ggc", "ig", "ingrad"]
    for method in methods:
        if method == "residual":
            method = "baseline"
        base_res = base_to_res[method]

        for attr, results in base_res.items():
            xx = [k*0.01 for k in range(101)]
            yy = []

            sample_to_xy = {k: [[],[]] for k in range(len(results[0]["mrf"]))}
            for thr in results:
                mask_sizes = [mask_size/(size**2) for mask_size in results[thr]["mask_size"]]
                mrfs = [mrf for mrf in results[thr]["mrf"]]

                for sample in range(len(mask_sizes)):
                    sample_to_xy[sample][0].append(mask_sizes[sample])
                    sample_to_xy[sample][1].append(mrfs[sample])

            for sample, dat in sample_to_xy.items():
                f = interp1d([1] + dat[0] + [0], [dat[1][0]] + dat[1] + [0])
                yy_sample = [f(x) for x in xx]
                yy.append(yy_sample)

            yy = np.mean(yy, axis=0)
            auc = np.trapz(yy, xx)
            print(method, attr, f"DAPI SCORE: {auc}")

            if attr == "D" or attr == "residual":
                color = method_to_color[method]
                label = "$\mathregular{D}$"
                linestyle="-"
                if attr == "residual":
                    method = "res"
                l, = plt.plot(xx,yy,color=color,
                              label="D-$\mathregular{" + f"{method.upper()}" + "}$ (ours)",
                              linewidth=2, alpha=1, linestyle=linestyle)
                plot_lines_dapi.append(l)

            else:
                continue

    l0, = plt.plot([0,0], [0,0], linestyle="-", alpha=1, color="black")
    l1, = plt.plot([0,0], [0,0], linestyle="--", alpha=1, color="black")

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xticks([0,0.5,1],[0,0.5,1])
    plt.yticks([0.5,1],[0.5,1])

def sort_results(res):
    base_methods = ["dl", "gc", "ggc", "ig", "ingrad"]
    base_to_res = {}
    for method in base_methods:
        base_to_res[method] = {"D": None,
                               "S": None}

    base_to_res["baseline"] = {"residual": None,
                               "random": None}

    for base_method in base_methods:
        base_to_res[base_method]["D"] = res["d_" + base_method]
        base_to_res[base_method]["S"] = res[base_method]

    base_to_res["baseline"]["residual"] = res["residual"]
    base_to_res["baseline"]["random"] = res["random"]

    return base_to_res

def parse_goyal_results(dataset_dir, classes, img_shape):
    methods_to_results = {}
    directions = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) if d[0] != "."]

    samples = []
    for direction in directions:
        from_to = os.path.basename(direction).split("_")
        query_index = classes.index(from_to[0])
        distractor_index = classes.index(from_to[1])

        samples_direction = [(os.path.join(direction, d, "predictions.csv"), (query_index, distractor_index)) for d
                             in os.listdir(direction) if d[0] != "."]
        samples.extend(samples_direction)

    # Loop over samples:
    sample_to_xy = {}

    for i in tqdm(range(len(samples))):
        sample = samples[i]
        path = sample[0]
        qd_ids = sample[1]

        csv_sample = []
        try:
            with open(path, "r") as f:
                content = csv.reader(f, delimiter=",")
                for line in content:
                    csv_sample.append(line)
        except:
            continue

        # Query Dist
        query_index = qd_ids[0]
        dist_index = qd_ids[1]

        # Header
        header = csv_sample[0]

        # Get number of classes:
        n_classes = len([s for s in header if "score" in s])

        # mask size column:
        mask_size_column = header.index("mask_size_px")
        query_column = header.index(f"score_{query_index}")
        dist_column = header.index(f"score_{dist_index}")

        # For each sample pair we need to renormalize
        # with the image score we copy to.
        # Here we copy to query, i.e. we need
        # to subtract f(query)[distractor_class]
        # First prediction is query,
        # Last is distractor
        normalization = float(csv_sample[1][dist_column])

        # We start at 1 as this is the zero point
        # We do not include -1 as this is the distractor
        dapi_scores = []
        mask_sizes = []
        for row in csv_sample[1:-1]:
            dapi_score = float(row[dist_column]) - normalization
            mask_size = float(row[mask_size_column])

            dapi_scores.append(dapi_score)
            mask_sizes.append(mask_size/np.prod(img_shape))

        sample_to_xy[i] = (mask_sizes, dapi_scores)

    return sample_to_xy

def plot_goyal(sample_to_xy, color, label, linestyle):
    xx = sample_to_xy[0][0]
    yy = []

    for sample, dat in sample_to_xy.items():
        f = interp1d([1] + dat[0] + [0], [dat[1][0]] + dat[1] + [0])
        yy_sample = [f(x) for x in xx]
        yy.append(yy_sample)

    yy = np.mean(yy, axis=0)
    auc = np.trapz(yy, xx)
    print(f"GOYAL DAPI SCORE: {auc}")

    l, = plt.plot(xx,yy,color=color, label=label, linewidth=2, alpha=1, linestyle=linestyle)
    plt.legend()

def get_auc(sample_to_xy):
    xx = sample_to_xy[0][0]
    yy = []
    for sample, dat in sample_to_xy.items():
        f = interp1d([1] + dat[0] + [0], [dat[1][0]] + dat[1] + [0])
        yy_sample = [f(x) for x in xx]
        yy.append(yy_sample)

    yy = np.mean(yy, axis=0)
    auc = np.trapz(yy, xx)
    return auc

def plot_dset(dset, goyal_path, dapi_path, net, method):
    dset_to_classes = {"summer_winter": ["summer", "winter"],
                       "horses_zebras": ["horses", "zebras"],
                       "apples_oranges": ["apples", "oranges"],
                       "disc_a": ["0", "1"],
                       "disc_b": ["0", "1", "2"],
                       "mnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                       "synapses": ["gaba", "acetylcholine", "glutamate", "serotonin",  "octopamine", "dopamine"]}

    dset_to_size = {"summer_winter": (256, 256),
                    "horses_zebras": (256,256),
                    "apples_oranges": (256,256),
                    "disc_a": (128,128),
                    "disc_b": (128,128),
                    "mnist": (28,28),
                    "synapses": (128,128)}

    classes = dset_to_classes[dset]
    sample_to_xy_goyal = parse_goyal_results(goyal_path, classes, dset_to_size[dset])
    sample_to_xy_dapi = parse_dapi_results(dapi_path)
    plot_dapi(sample_to_xy_dapi, dset, net="vgg", leg=True, methods=[method])
    plot_goyal(sample_to_xy_goyal, "black", "Goyal et al.", "-")
    plt.grid()
    plt.savefig(dset + f"_goyal_{net}.png", bbox_inches = 'tight', pad_inches = 0)
    plt.clf()


if __name__ == "__main__":
    args = parser.parse_args()
    plot_dset(args.dset, args.goyal_dset_path, args.dapi_dset_path, args.net, args.dapi_method)
