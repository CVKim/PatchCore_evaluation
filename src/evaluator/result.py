import os
import numpy as np
import pandas as pd

from . import th_tuner, metrics, on_blobs
from .. import utils


def save_result(
    image_paths,
    image_scores: np.ndarray,
    labels_gt,
    score_masks: np.ndarray,
    masks_gt,
    test_data_name,
    save_dir_path,
    num_ths=21,
    min_size=60,
):
    
    th_max_iou, max_iou, ths, ious = th_tuner.tune_score_threshold_to_maximize_iou(
        masks_gt,
        score_masks,
        num_ths=num_ths
    )
    
    th_max_iou_ms, max_iou_ms, ths_ms, _ = th_tuner.tune_score_threshold_to_maximize_iou(
        masks_gt,
        score_masks,
        num_ths=num_ths,
        min_size=min_size,
        score_thresholds=ths + [th_max_iou]
    )

    binarized_score_masks_th_max_iou = np.array(score_masks > th_max_iou).astype(np.uint8)
    binarized_score_masks_th_max_iou_ms = on_blobs.remove_small_blobs(np.array(score_masks > th_max_iou_ms).astype(np.uint8), min_size)

    utils.plot_hist(
        score_masks,
        masks_gt,
        filename=os.path.join(save_dir_path, "hist_pixel.png"),
        other_points={"th_max_iou": th_max_iou, 
                      "th_max_iou_ms": th_max_iou_ms
                      },
    )

    utils.plot_score_masks(
        save_dir_path=f"{save_dir_path}/plot_raw/{test_data_name}_th_max_iou",
        image_paths=image_paths,
        masks_gt=masks_gt,
        score_masks=score_masks,
        image_scores=image_scores,
        binary_masks=binarized_score_masks_th_max_iou
    )

    utils.plot_score_masks(
        save_dir_path=f"{save_dir_path}/plot_processed/{test_data_name}_th_max_iou",
        image_paths=image_paths,
        masks_gt=masks_gt,
        score_masks=score_masks,
        image_scores=image_scores,
        binary_masks=binarized_score_masks_th_max_iou_ms
    )

    for th in ths_ms:
        binary_masks = on_blobs.remove_small_blobs(np.array(score_masks > th).astype(np.float32), min_size)
        utils.plot_score_masks(
            save_dir_path=f"{save_dir_path}/plot_processed/{test_data_name}_th{th:.3f}",
            image_paths=image_paths,
            masks_gt=masks_gt,
            score_masks=score_masks,
            image_scores=image_scores,
            binary_masks=binary_masks
        )

    try:
        print("Computing image auroc...")
        image_auroc = metrics.compute_imagewise_retrieval_metrics(
            image_scores, labels_gt
        )["auroc"]
    except:
        image_auroc = 0.
        print("Failed at computing image auroc...")
    # print("Computing pixel auroc...")
    # pixel_auroc = metrics.compute_pixelwise_retrieval_metrics(
    #     score_masks, masks_gt
    # )["auroc"]

    result = {"test_data_name": test_data_name}
    result["max_iou"] = max_iou
    result["image_auroc"] = image_auroc

    return result



def summarize_result(result_list, save_dir_path):
    df = pd.DataFrame(result_list)

    # Save to CSV
    save_path = os.path.join(save_dir_path, 'result.csv')
    df.to_csv(save_path, index=False)  # 'index=False' to avoid writing row numbers

    return df