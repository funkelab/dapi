from datasets import GetDataset
from solvers import ContinuousSolver, BatchedExhaustiveSolver
from tqdm import tqdm
from utils import (
    create_hybrid_sequence,
    save_image_sequnce,
    extract_features,
    load_pretrained_model_weights,
    save_q_and_d,
)
import numpy as np
import torch
import pandas as pd
from pathlib2 import Path
import configparser
import json
import argparse
import os


def find_edits(query_image, distractor_image, distractor_label, model, solver):
    '''Takes a query image, a distractor image, and the label of the distractor image. Returns a list of edit steps to create a sequence of hybrid images, where the last ones are classified as distractor labels. This is an implementation of Algorithm 1 from Goyal Paper. Solvers BestEdit, which can be exhaustive or continues. 
    
    '''
    query_features = extract_features(query_image, model)
    distractor_features = extract_features(distractor_image, model)

    num_patches = query_features.shape[-1]

    edits = []
    feature_prediction = []

    for i in tqdm(range(num_patches)):

        edit, best_feature_prediction = solver.best_edit(
            query_features, distractor_features, distractor_label, edits, model
        )

        edits.append(edit)
        feature_prediction.append(best_feature_prediction)

        # copy features from distractor to query
        query_features = apply_edit(query_features, distractor_features, edit)

        # for debugging
        # break

    return num_patches, edits, feature_prediction


def predict_and_csv(
    hybrid_sequnce,
    model,
    path,
    num_patches,
    num_classes,
    image_size,
    feature_prediction,
):
    # getting prediction
    fw = np.sqrt(num_patches)
    patch_size = (image_size // fw) ** 2

    hybrid_sequence = torch.stack(hybrid_sequnce)
    # num_patches, h w, -> num_patches, ch, h w
    # hybrid_sequence = hybrid_sequence
    with torch.no_grad():
        y_hat = model(hybrid_sequence.cuda()).detach().cpu()
        y_hat = torch.softmax(y_hat, 1).numpy()
    df = pd.DataFrame(y_hat)
    df.columns = [f"score_{i}" for i in range(num_classes)]
    # our mask size is just num_patches * position
    # np.arange starts with 0, in order to avoid I will just add 1 to sequnce
    df["mask_size_px"] = np.arange(y_hat.shape[0]) * patch_size
    df["mask_index"] = np.arange(y_hat.shape[0])
    df["prediction"] = np.argmax(y_hat, 1)
    df["num_patchs"] = num_patches
    df["patch_size"] = patch_size

    # we will insert predicton for our original image
    # and prediction for our last images (distractor)

    feature_prediction.insert(0, y_hat[0].copy())
    feature_prediction.append(y_hat[-1].copy())
    df_features = pd.DataFrame(feature_prediction)
    df_features.columns = [f"ft_score_{i}" for i in range(num_classes)]
    df_features["feature_prediction"] = np.stack(feature_prediction).argmax(1)

    df = pd.concat([df_features, df], axis=1)
    df.to_csv(f"{path}/predictions.csv", index=False)


def apply_edit(query_features, distractor_features, edit):

    copy_from, copy_to = edit

    tqdm.write(f"Copying from {copy_from} to {copy_to}")
    tqdm.write(f"")

    # query_features: (c, n)
    # distractor_features: (c, n)

    # query_features = query_features.clone()
    query_features[:, copy_to] = distractor_features[:, copy_from]
    return query_features


if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        help="select model type look at config files .ini on top",
        type=str,
    )
    parser.add_argument("--model_name", help="PyTorch Model Class name", type=str)
    parser.add_argument(
        "--dataset_name",
        help="Dataset Name, will use for creating of folders",
        type=str,
    )
    parser.add_argument(
        "--class_names",
        help="class names from-->to, e.g glutamate_acetylcholine, horses_zebras",
        type=str,
    )
    parser.add_argument(
        "--query_image_list",
        help="text file names from that used for calculating all the DAPI scores",
        type=str,
    )
    parser.add_argument(
        "--folder_name", help="folder name where data is located", type=str
    )
    parser.add_argument("--config_name", help="config file names", type=str)
    parser.add_argument("--chk_name", help="model checkpoint", type=str)
    parser.add_argument(
        "--distractor_folder",
        help="raw files folder for getting distractor images",
        type=str,
    )
    parser.add_argument(
        "--flatten", help="if 1 channel", type=eval, choices=[True, False], default=True
    )
    parser.add_argument("--solvers", help="exhaustive or continues", type=str, default="exhaustive")
    args = parser.parse_args()

    model_type = args.model_type
    model_name = args.model_name
    dataset_name = args.dataset_name
    class_names = args.class_names
    txt_fln = args.query_image_list
    folder_name = args.folder_name
    config_name = args.config_name
    chk_name = args.chk_name
    raw_files_folder = args.distractor_folder
    flatten = args.flatten
    solvers = args.solvers

    if solvers == "exhaustive":
        print("using exhaustive solver")
        solver = BatchedExhaustiveSolver()

    if solvers == "continues":
        print("using continues solver")
        solver = ContinuousSolver(100)

    dataset = GetDataset(
        class_names=class_names,
        model_type=model_type,
        flatten=flatten,
        txt_fln=txt_fln,
        folder=folder_name,
        config_name=config_name,
        raw_files_folder=raw_files_folder,
    )

    config = configparser.ConfigParser()
    config.read(config_name)
    num_classes = eval(config[model_type]["output_classes"])
    model = load_pretrained_model_weights(
        model_name=model_name,
        chk_name=chk_name,
        downsample_factors=[
            eval(i) for i in (config[model_type]["downsample_factors"]).split(";")
        ],
        input_size=eval(config[model_type]["input_shape"]),
        input_channels=eval(config[model_type]["channels"]),
        output_classes=eval(config[model_type]["output_classes"]),
    )

    model.cuda()
    model.eval()
    model.requires_grad_ = False
    for (i, items) in enumerate(dataset):
        (
            query_image,
            query_label,
            distractor_image,
            distractor_label,
            query_image_name,
            distractor_image_name,
        ) = items
        print(i)
        tqdm.write(
            f"Creating sequences for query={query_label}, " f"target={distractor_label}"
        )

        store_path = f"data/{model_type}/{dataset_name}/{class_names}/{i}_{query_image_name.name[:-4]}->{distractor_image_name.name[:-4]}/"

        if os.path.exists(store_path + "predictions.csv"):
            continue

        # storing orginal images
        save_q_and_d(store_path, q=query_image, d=distractor_image)

        query_image = torch.tensor(query_image)
        distractor_image = torch.tensor(distractor_image)

        num_patches, edits, feature_prediction = find_edits(
            query_image, distractor_image, distractor_label, model, solver
        )
        print(num_patches)

        with open(f"{store_path}edits.json", "w") as f:
            json.dump(edits, f)

        hybrid_sequence = create_hybrid_sequence(
            query_image, distractor_image, edits, num_patches=num_patches
        )

        # adding distractor at the end
        # adding query image in the begining
        hybrid_sequence.insert(0, query_image)
        hybrid_sequence.append(distractor_image)

        save_image_sequnce(store_path, hybrid_sequence)
        predict_and_csv(
            hybrid_sequnce=hybrid_sequence,
            model=model,
            path=store_path,
            num_patches=num_patches,
            num_classes=num_classes,
            image_size=query_image.shape[-1],
            feature_prediction=feature_prediction,
        )