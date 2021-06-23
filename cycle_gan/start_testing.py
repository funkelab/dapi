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
                  submit_cmd="python -u"):

    # Workaround for cycle_gan convention
    name = os.path.basename(checkpoints_dir)
    checkpoints_dir = os.path.dirname(checkpoints_dir)

    sub = f"{submit_cmd} cycle_gan/test.py"
    base_cmd = sub + " --model test --no_dropout --results_dir {} --dataroot {} "+\
                     "--checkpoints_dir {} --name {} --model_suffix {} --num_test {} "+\
                     "--aux_checkpoint {} --aux_input_size {} --aux_net {} --aux_input_nc 1 "+\
                     "--num_threads 1 --verbose --aux_output_classes {} --aux_downsample_factors '{}'"

    if test_class == class_pair[0]:
        a_or_b = "A"
    elif test_class == class_pair[1]:
        a_or_b = "B"

    data_root = os.path.join(data_root, "train" + a_or_b)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    aux_downsample_factors_string = ""
    for fac in aux_downsample_factors:
        for dim in fac:
            aux_downsample_factors_string+=f"{dim},"
        aux_downsample_factors_string=aux_downsample_factors_string[:-1] + "x"
    aux_downsample_factors_string = aux_downsample_factors_string[:-1]

    cmd = base_cmd.format(results_dir,
                          data_root,
                          checkpoints_dir,
                          name,
                          "_" + a_or_b,
                          num_test,
                          aux_checkpoint,
                          input_size,
                          aux_net,
                          aux_output_classes,
                          aux_downsample_factors_string)

    subprocess.Popen(cmd,
                     shell=True)

def test_experiment(data_root,
                    gan_checkpoint_dir,
                    aux_net,
                    aux_checkpoint,
                    aux_downsample_factors,
                    classes,
                    input_size,
                    num_test,
                    submit_cmd):

    class_pairs = [[i,j] for i,j in list(itertools.combinations(classes, 2))]
    for class_pair in class_pairs:
        dataset = f"{class_pair[0]}_{class_pair[1]}"
        data_dir = f"{data_root}/{dataset}"
        gan_checkpoint = f"{gan_checkpoint_dir}/{dataset}"
        aux_output_classes = len(classes)
        
        if aux_net == "vgg":
            aux_net = "vgg2d"

        for test_class in class_pair:
            results_dir = os.path.join(data_root, f"results/{aux_net}_{test_class}")
            start_testing(class_pair,
                          test_class,
                          gan_checkpoint,
                          data_dir,
                          results_dir,
                          aux_checkpoint,
                          aux_output_classes,
                          aux_downsample_factors=aux_downsample_factors,
                          aux_net=aux_net,
                          input_size=input_size,
                          netG="resnet_9blocks",
                          num_test=num_test,
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
                          "synapses": {"classes": ["gaba", "acetylcholine", "glutamate","serotonin", "octopamine", "dopamine"],
                                       "input_size": 128,
                                       "num_test": 500,
                                       "aux_downsample_factors": [(2,2),(2,2),(2,2),(2,2)]}}

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

    test_experiment(data_root,
                    gan_checkpoint_dir,
                    aux_net,
                    aux_checkpoint,
                    aux_downsample_factors,
                    classes,
                    input_size,
                    num_test,
                    submit_cmd)
