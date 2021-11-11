import os
import subprocess
import itertools
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--aux_net", type=str, required=True, help="'vgg' or 'res'")
parser.add_argument("--aux_checkpoint", type=str, required=True)
parser.add_argument("--gan_checkpoint_dir", type=str, required=True)
parser.add_argument("--submit_cmd", type=str, required=False, default="python -u")

def start_testing(class_pair,
                  test_class,
                  checkpoints_dir,
                  data_root,
                  results_dir,
                  aux_checkpoint,
                  aux_output_classes,
                  aux_downsample_factors=[(2,2),(2,2),(2,2),(2,2)],
                  aux_net="vgg2d",
                  input_size=128,
                  netG="resnet_9blocks",
                  num_test=500,
                  preprocess=None,
                  load_size=None,
                  crop_size=None,
                  input_nc=1,
                  output_nc=1,
                  submit_cmd="python -u"):

    if aux_net == "vgg":
        aux_net = "vgg2d"

    # Workaround for cycle_gan convention
    name = os.path.basename(checkpoints_dir)
    checkpoints_dir = os.path.dirname(checkpoints_dir)

    # Use directional checkpoint and dataset
    if test_class == class_pair[0]:
        source = "A"
    elif test_class == class_pair[1]:
        source = "B"
    else:
        raise ValueError("test_class or class_pair corrupted")

    # checkpoints are named based on their source:
    model_suffix = "_" + source

    # choose corresponding dataset
    data_root = os.path.join(data_root, "train"+source)

    # Make results dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Base command
    sub = f"{submit_cmd} cycle_gan/test.py "
    base_cmd = sub + "--model test "+\
                     "--no_dropout "+\
                     "--verbose "+\
                     f"--results_dir {results_dir} "+\
                     f"--dataroot {data_root} "+\
                     f"--checkpoints_dir {checkpoints_dir} "+\
                     f"--name {name} "+\
                     f"--num_test {num_test} "+\
                     f"--netG {netG} "+\
                     f"--num_threads 1 "+\
                     f"--input_nc {input_nc} "+\
                     f"--output_nc {output_nc} "+\
                     f"--model_suffix {model_suffix} "
   
    # preprocess options
    if preprocess is not None:
        base_cmd += f"--preprocess {preprocess} "
        if load_size is not None:
            base_cmd += f"--load_size {load_size} "
        if crop_size is not None:
            base_cmd += f"--crop_size {crop_size} "

    # Aux net options
    if aux_net is not None:
        aux_downsample_factors_string = ""
        for fac in aux_downsample_factors:
            for dim in fac:
                aux_downsample_factors_string+=f"{dim},"
            aux_downsample_factors_string=aux_downsample_factors_string[:-1] + "x"
        aux_downsample_factors_string = aux_downsample_factors_string[:-1]

        aux_cmd = f"--aux_checkpoint {aux_checkpoint} "+\
                  f"--aux_input_size {input_size} "+\
                  f"--aux_net {aux_net} "+\
                  f"--aux_input_nc {input_nc} "+\
                  f"--aux_output_classes {aux_output_classes} "+\
                  f"--aux_downsample_factors '{aux_downsample_factors_string}' "

        base_cmd += aux_cmd

    subprocess.Popen(base_cmd,
                     shell=True)

def test_experiment(data_root,
                    gan_checkpoint_dir,
                    aux_net,
                    aux_checkpoint,
                    aux_downsample_factors,
                    classes,
                    input_size,
                    num_test,
                    netG="resnet_9blocks",
                    preprocess=None,
                    load_size=None,
                    crop_size=None,
                    input_nc=1,
                    output_nc=1,
                    submit_cmd="python -u"):

    class_pairs = [[i,j] for i,j in list(itertools.combinations(classes, 2))]
    for class_pair in class_pairs:
        dataset = f"{class_pair[0]}_{class_pair[1]}"
        data_root = f"{data_root}/{dataset}"
        gan_checkpoint_dir = f"{gan_checkpoint_dir}/{dataset}"
        aux_output_classes = len(classes)
        
        if aux_net == "vgg":
            aux_net = "vgg2d"

        for test_class in class_pair:
            results_dir = os.path.join(data_root, f"results/{aux_net}_{test_class}")
            start_testing(class_pair=class_pair,
                          test_class=test_class,
                          checkpoints_dir=gan_checkpoint_dir,
                          data_root=data_root,
                          results_dir=results_dir,
                          aux_checkpoint=aux_checkpoint,
                          aux_output_classes=aux_output_classes,
                          aux_downsample_factors=aux_downsample_factors,
                          aux_net=aux_net,
                          input_size=input_size,
                          netG=netG,
                          num_test=num_test,
                          preprocess=preprocess,
                          load_size=load_size,
                          crop_size=crop_size,
                          input_nc=input_nc,
                          output_nc=output_nc,
                          submit_cmd=submit_cmd)


def get_config(experiment):
    experiment_to_cfg = {"mnist": {"classes": [k for k in range(10)],
                                   "input_size": 28,
                                   "num_test": 1000,
                                   "aux_downsample_factors": [(2,2),(2,2),(1,1),(1,1)]},
                          "disc_a": {"classes": [0,1],
                                     "input_size": 128,
                                     "num_test": 1000,
                                     "aux_downsample_factors": [(2,2),(2,2),(2,2),(2,2)]},
                          "disc_b": {"classes": [0,1,2],
                                     "input_size": 128,
                                     "num_test": 1000,
                                     "aux_downsample_factors": [(2,2),(2,2),(2,2),(2,2)]},
                          "synapses": {"classes": ["gaba", "acetylcholine", "glutamate",
                                                   "serotonin", "octopamine", "dopamine"],
                                       "input_size": 128,
                                       "num_test": 500,
                                       "aux_downsample_factors": [(2,2),(2,2),(2,2),(2,2)]},
                          "horses_zebras": {"classes": ["horses", "zebras"],
                                            "input_size": 256,
                                            "aux_downsample_factors": [(2,2),(2,2),(2,2),(2,2)],
                                            "num_test": 500,
                                            "preprocess": "resize_and_crop",
                                            "load_size": 286,
                                            "crop_size": 256,
                                            "input_nc": 3,
                                            "output_nc": 3},
                          "apples_oranges": {"classes": ["apples", "oranges"],
                                            "input_size": 256,
                                            "aux_downsample_factors": [(2,2),(2,2),(2,2),(2,2)],
                                            "num_test": 500,
                                            "preprocess": "resize_and_crop",
                                            "load_size": 286,
                                            "crop_size": 256,
                                            "input_nc": 3,
                                            "output_nc": 3},
                         "summer_winter": {"classes": ["summer", "winter"],
                                            "input_size": 256,
                                            "aux_downsample_factors": [(2,2),(2,2),(2,2),(2,2)],
                                            "num_test": 500,
                                            "preprocess": "resize_and_crop",
                                            "load_size": 286,
                                            "crop_size": 256,
                                            "input_nc": 3,
                                            "output_nc": 3} }

    return experiment_to_cfg[experiment]

if __name__ == "__main__":
    args = parser.parse_args()
    exp_cfg = get_config(args.experiment)
    data_root = args.data_root
    gan_checkpoint_dir = args.gan_checkpoint_dir
    submit_cmd = args.submit_cmd
    aux_net = args.aux_net
    aux_checkpoint = args.aux_checkpoint
    aux_downsample_factors = exp_cfg["aux_downsample_factors"]
    classes = exp_cfg["classes"]
    num_test = exp_cfg["num_test"]
    input_size = exp_cfg["input_size"]

    if "preprocess" in exp_cfg:
        preprocess = exp_cfg["preprocess"]
        load_size = exp_cfg["load_size"]
        crop_size = exp_cfg["crop_size"]
    else:
        preprocess = None
        load_size = None
        crop_size = None

    if "input_nc" in exp_cfg:
        input_nc = exp_cfg["input_nc"]
        output_nc = exp_cfg["output_nc"]

    else:
        input_nc = 1
        output_nc = 1

    test_experiment(data_root,
                    gan_checkpoint_dir,
                    aux_net,
                    aux_checkpoint,
                    aux_downsample_factors,
                    classes,
                    input_size,
                    num_test,
                    netG="resnet_9blocks",
                    preprocess=preprocess,
                    load_size=load_size,
                    crop_size=crop_size,
                    input_nc=input_nc,
                    output_nc=output_nc,
                    submit_cmd=submit_cmd)
