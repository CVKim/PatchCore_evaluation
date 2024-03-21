import csv
import logging
import os
import random
import tqdm
import yaml
import cv2
import PIL
import torch
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=2,
    # dataset = None
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    # '''
    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        # if not isinstance(image, np.ndarray):
        # image = image.numpy()
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        # if masks_provided:
        try:
            mask = PIL.Image.open(mask_path).convert("RGB")
            mask = mask_transform(mask)
            if not isinstance(mask, np.ndarray):
                mask = mask.numpy()
        except:
            mask = np.zeros_like(image)
        # '''
        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        figure, axes = plt.subplots(1, 3)
        axes[0].imshow(image)
        # axes[2].imshow(segmentation)
        axes[1].imshow(image, alpha=1)
        axes[1].imshow(segmentation, alpha=0.5)
        axes[2].imshow(mask)
        figure.suptitle("Anomaly Score: {:.3f}".format(anomaly_score))
        figure.set_size_inches(3 * (2 + int(masks_provided)), 3)
        figure.tight_layout()
        figure.savefig(savename, dpi=300)
        plt.close()


def plot_score_masks(
    save_dir_path, image_paths, masks_gt, score_masks, image_scores, binary_masks, overlay=True,
):
    os.makedirs(save_dir_path, exist_ok=True)

    for i, (image_path, mask_gt, score_mask, image_score) in tqdm.tqdm(
        enumerate(zip(image_paths, masks_gt, score_masks, image_scores)),
        total=len(image_paths),
        desc="Plotting score masks...",
        leave=False,
    ):
        base_filename = "_".join(image_path.split("/")[-2:])
        base_filename_without_ext = os.path.splitext(base_filename)[0]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        savename = os.path.join(save_dir_path, base_filename_without_ext + ".jpg")

        
        resized_image = cv2.resize(image, (mask_gt.shape[1], mask_gt.shape[0]))

        
        heatmap = cv2.applyColorMap(
            np.uint8(255 * score_mask), cv2.COLORMAP_JET
        )  # cv2.COLORMAP_JET cv2.COLORMAP_HOT
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Convert mask_gt for visualization
        mask_gt_vis = (mask_gt * 255).astype(np.uint8)
        mask_gt_vis = cv2.cvtColor(mask_gt_vis, cv2.COLOR_GRAY2RGB)
        mask_gt_vis[:, :, 1:] = 0 

        if binary_masks is not None:
            binary_mask = (binary_masks[i] * 255).astype(np.uint8)
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
            binary_mask[:, :, 1:] = 0 
        else:
            binary_mask = heatmap

        if overlay:
            heatmap = cv2.addWeighted(resized_image, 0.2, heatmap, 0.8, 0)
            mask_gt_vis = cv2.addWeighted(resized_image, 0.5, mask_gt_vis, 0.5, 0)
            binary_mask = cv2.addWeighted(resized_image, 0.5, binary_mask, 0.5, 0)

        # Combine resized main image, heatmap, and mask_gt side by side
        combined_image = np.hstack((resized_image, heatmap, mask_gt_vis))

        top_row = np.hstack((resized_image, heatmap))
        bottom_row = np.hstack((mask_gt_vis, binary_mask))

        # Combine the two rows to get a 2x2 grid
        combined_image = np.vstack((top_row, bottom_row))

        cv2.imwrite(savename, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids) > 0:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=None,
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if column_names is None:
        column_names = [
            "Instance AUROC",
            "Full Pixel AUROC",
            "Full PRO",
            "Anomaly Pixel AUROC",
            "Anomaly PRO",
        ]
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics


# Added.


def dirs(args):
    data_config, data_seed = args.data_path.split("/")[-2:]
    coreset_config = "_".join(data_config.split("_")[:-1])
    data_seed = "".join([c for c in data_seed if c.isdigit()])

    coreset_dirs = []
    for (
        stype,
        gratio,
        gdim,
        lofT,
        lofk,
        softweight,
        kthnndnorm,
        kthnndk,
        kthnndp,
        kthnndT,
        cdboundp,
        cdboundT,
        cddetectp,
        cddetectT,
    ) in zip(
        args.sampler_type,
        args.greedy_ratio,
        args.greedy_proj_dim,
        args.lof_thresh,
        args.lof_k,
        args.weight,
        args.normalize,
        args.kthnnd_k,
        args.kthnnd_p,
        args.kthnnd_T,
        args.cd_bound_p,
        args.cd_bound_T,
        args.cd_detect_p,
        args.cd_detect_T,
    ):
        coreset_dirs.append(
            f"./coreset/{coreset_config}"
            + f"/{stype}/GREEDY_per{gratio}_dim{gdim}/LOF_thresh{lofT}_k{lofk}_softweight{softweight}"
            + f"/KTHD_norm{kthnndnorm}_k{kthnndk}_bootp{kthnndp}_bootT{kthnndT}"
            + f"/CDBOUND_bootp{cdboundp}_bootT{cdboundT}/CDDETECT_bootp{cddetectp}_bootT{cddetectT}"
            + f"/FAISS_metric{args.faiss_distance}_k{args.faiss_k_neighbor}/SEED_dat{data_seed}_mod{args.model_seed}"
        )

    result_dir = (
        f"./result/{data_config}"
        + f"/{args.sampler_type}/GREEDY_per{args.greedy_ratio}_dim{args.greedy_proj_dim}/LOF_thresh{args.lof_thresh}_k{args.lof_k}_softweight{args.weight}"
        + f"/KTHD_norm{args.normalize}_k{args.kthnnd_k}_bootp{args.kthnnd_p}_bootT{args.kthnnd_T}"
        + f"/CDBOUND_bootp{args.cd_bound_p}_bootT{args.cd_bound_T}/CDDETECT_bootp{args.cd_detect_p}_bootT{args.cd_detect_T}"
        + f"/FAISS_metric{args.faiss_distance}_k{args.faiss_k_neighbor}/SEED_dat{data_seed}_mod{args.model_seed}"
    )

    return coreset_dirs, result_dir


def clists():
    mvtec_clist = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    visa_clist = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]

    return mvtec_clist, visa_clist


def minmax_normalize_image_scores(image_scores):
    scores = image_scores
    min_scores = scores.min(axis=-1).reshape(-1, 1)
    max_scores = scores.max(axis=-1).reshape(-1, 1)
    scores = (scores - min_scores) / (max_scores - min_scores + 1e-5)
    scores = np.mean(scores, axis=0)

    return scores


def minmax_normalize_score_masks(score_masks):
    segmentations = score_masks
    min_scores = (
        segmentations.reshape(len(segmentations), -1).min(axis=-1).reshape(-1, 1, 1, 1)
    )
    max_scores = (
        segmentations.reshape(len(segmentations), -1).max(axis=-1).reshape(-1, 1, 1, 1)
    )
    segmentations = (segmentations - min_scores) / (max_scores - min_scores + 1e-5)
    segmentations = np.mean(segmentations, axis=0)  # means for ensembled coreset!

    return segmentations


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_folder_names(folder_root_path):
    # List to store folder names
    folder_names = []

    # Check if the provided path is indeed a directory
    if not os.path.isdir(folder_root_path):
        return "The provided path is not a directory."

    # Iterate over the entries in the given directory
    for entry in os.listdir(folder_root_path):
        # Full path of the entry
        full_path = os.path.join(folder_root_path, entry)

        # Check if the entry is a directory
        if os.path.isdir(full_path):
            folder_names.append(entry)

    # Sort the folder names in alphabetical order
    return sorted(folder_names)


import yaml
import argparse


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_config_args(config_path):
    return Namespace(**load_yaml(config_path))


class Namespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = Namespace(**value)
            setattr(self, key, value)


def plot_hist(
    scores, labels=None, filename="./histogram.png", bin_count=50, other_points=None
):
    scores = np.array(scores).flatten()

    if labels is not None:
        labels = np.array(labels).astype(np.uint8).flatten()
        scores_label_0 = scores[labels == 0]
        scores_label_1 = scores[labels == 1]

        # Efficient histogram plotting
        plt.hist(
            [scores_label_0, scores_label_1],
            bins=bin_count,
            alpha=0.5,
            label=["normal", "defect"],
        )
        plt.legend()
    else:
        plt.hist(scores, bins=bin_count, alpha=0.5)

    # Basic labels
    plt.xlabel("Scores")
    plt.ylabel("Frequency")

    # Plot additional points if provided
    if other_points:
        for label, point in other_points.items():
            plt.axvline(x=point, color="k", linestyle="--", alpha=0.7)
            plt.text(
                point,
                plt.gca().get_ylim()[1],
                f" {label}",
                rotation=90,
                verticalalignment="top",
            )

    # Save the plot as a PNG file
    plt.savefig(filename, bbox_inches="tight")

    # Clear the current plot to free memory
    plt.clf()

def plot_scores(scores, filename="./scores.png", title="1D Scores Plot", x_label="Index", y_label="Score"):
    # Create the figure
    # plt.figure()

    # Plotting logic
    plt.plot(scores, marker='o', alpha=0.25, markersize=0.25)  # 'o' is for circular markers
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    plt.ylim(-0.05,+0.05 )
    plt.xlim(0, 250)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save and close the figure
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()
    plt.close()


def save_dict(data, file_path):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file:
        pkl.dump(data, file)


def load_dict(file_path):
    with open(file_path, "rb") as file:
        return pkl.load(file)

from scipy.interpolate import InterpolatedUnivariateSpline
def compute_derivative(scores):
    """
    Computes the derivative of a given 1D numpy array 'scores' 
    with respect to its indices.

    Parameters:
    scores (numpy.ndarray): 1D array of scores.

    Returns:
    numpy.ndarray: The derivative of scores with respect to its indices.
    """
    # Generate x values as indices of the scores
    x = np.arange(len(scores))

    # Using cubic spline interpolation
    spline = InterpolatedUnivariateSpline(x, scores, k=3)

    # Derivative of the spline
    derivative = spline.derivative()(x)

    return derivative