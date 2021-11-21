import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, required=True)
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--net", type=str, required=True)
parser.add_argument("--no_leg", required=False, action="store_false")


mpl.rcParams['figure.dpi'] = 300
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)
mpl.rc('text', usetex=False)
mpl.rc('axes', linewidth=2)
mpl.rc('font', weight='bold')

def plot_dapi(base_to_res, experiment, net, leg=True):
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
                       "ingrad": "purple"}

    plot_lines_dapi = []
    plot_lines_base = []
    for method in ["dl", "gc", "ggc", "ig", "ingrad"]:
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
            print(method, attr, f"AUC: {auc}")

            if attr == "D":
                color = method_to_color[method]
                label = "$\mathregular{D}$"
                linestyle="-"

            elif attr == "S":
                color = "blue"
                color = method_to_color[method]
                label = "$\mathregular{0_r}$"
                linestyle="--"
            else:
                continue

            if attr == "D":
                l, = plt.plot(xx,yy,color=color, label=method, linewidth=2, alpha=1, linestyle=linestyle)
                plot_lines_dapi.append(l)
            else:
                l, = plt.plot(xx,yy,color=color, linewidth=2, alpha=1, linestyle=linestyle)
                plot_lines_base.append(l)

    l0, = plt.plot([0,0], [0,0], linestyle="-", alpha=1, color="black")
    l1, = plt.plot([0,0], [0,0], linestyle="--", alpha=1, color="black")

    for method in ["baseline"]:
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
            print(method, attr, f"AUC: {auc}")
            if attr == "random":
                lr, = plt.plot(xx,yy,color="red", linewidth=2, alpha=1, linestyle="dotted")
            else:
                lre, = plt.plot(xx,yy,color="black", linewidth=2, alpha=1, linestyle="dotted")

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal', adjustable='box')

    if leg:
        legend1 = plt.legend([l0, l1], ["D", "S"], bbox_to_anchor=(1.1,1))
        plt.legend(plot_lines_dapi + [lr, lre], ["DL", "GC", "GGC", "IG", "INGRAD", "RANDOM", "RESIDUAL"],bbox_to_anchor=(1.5,0.55))
        plt.gca().add_artist(legend1)
    plt.xticks([0,0.5,1],[0,0.5,1])
    plt.yticks([0.5,1],[0.5,1])
    plt.savefig(out_path, bbox_inches = 'tight', pad_inches = 0)

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


def parse_results(res_dir):
    methods_to_results = {}
    img_dirs = [os.path.join(res_dir, d) for d in os.listdir(res_dir) if not "." in d]
    for img_dir in img_dirs:
        methods = os.listdir(img_dir)
        methods_dirs = [os.path.join(img_dir, d) for d in methods if not "." in d]
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


def get_auc(results_dict, size=128, method_map={}):
    method_to_auc = {}
    #for method, results in results_dict.items():
    for method_name_old, method_name_new in method_map.items():
        results = results_dict[method_name_old]
        base_method = method_name_old.split("_")[0]

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

        method_to_auc[method_name_new] = np.trapz(yy, xx)
    return method_to_auc

if __name__ == "__main__":
    args = parser.parse_args()
    res_dir = args.result_dir
    experiment = args.experiment
    net = args.net
    res = parse_results(res_dir)
    plot_dapi(res, experiment, net, leg=args.no_leg)
