import numpy as np
from sklearn.metrics import roc_curve
from typing import Tuple

# from .on_blobs import compute_metrics, compute_average_iou
from . import on_blobs


def tune_score_threshold_to_minimize_fnfp(
    masks_gt,
    score_masks,
    iou_threshold=0.5,
) -> Tuple[dict, dict]:
    masks_gt = np.array(masks_gt).astype(np.uint8)
    score_masks = np.array(score_masks)

    assert len(score_masks) == len(masks_gt)

    min_score = np.min(score_masks)
    max_score = np.max(score_masks)
    score_thresholds = np.around(
        np.linspace(min_score, max_score, 21), 2
    )  # FIXME: consider only the mask_gt == 1

    # Initialize variables to store the optimal values
    best_threshold = None
    min_FN = float("inf")
    min_FP = float("inf")

    for threshold in score_thresholds:
        TP, TN, FP, FN = on_blobs.compute_metrics(
            masks_gt, score_masks, threshold, iou_threhold=iou_threshold
        )
        print(f"th: {threshold} TP: {TP} TN: {TN} FP: {FP} FN: {FN}")

        # Check if this threshold has the minimal FN so far
        if FN < min_FN:
            min_FN = FN
            min_FP = FP
            best_threshold = threshold
        # If FN is the same, check for minimal FP
        elif FN == min_FN and FP < min_FP:
            min_FP = FP
            best_threshold = threshold

    return {"th_at_minimal_fnfp": best_threshold}, {"min_FP": min_FP, "min_FN": min_FN}


def tune_score_threshold_to_maximize_iou(
    masks_gt, score_masks, num_ths, min_size: int = None, score_thresholds=None
) -> Tuple[dict, dict]:
    masks_gt = np.array(masks_gt).astype(np.uint8)
    score_masks = np.array(score_masks)

    assert len(masks_gt) == len(score_masks)
    if score_masks.ndim == 3:
        num_classes = 1
    elif score_masks.ndim == 4:
        num_classes = score_masks.shape[-1] - 1
        raise NotImplementedError()
    else:
        raise ValueError()

    if score_thresholds is None:
        min_score = np.min(score_masks)
        max_score = np.max(score_masks)
        score_thresholds = np.around(np.linspace(min_score, max_score, num_ths), 3)[
            1:-1
        ]

    # Initialize variables to store the optimal values
    best_threshold = 0
    best_average_iou = 0

    ths = []
    ious = []

    for threshold in score_thresholds:
        binarized_score_masks = (score_masks >= threshold).astype(np.uint8)
        if min_size is not None:
            assert isinstance(min_size, int)
            binarized_score_masks = on_blobs.remove_small_blobs(
                binarized_score_masks, min_size
            )
        average_iou = on_blobs.compute_average_iou(masks_gt, binarized_score_masks)

        if average_iou > best_average_iou:
            best_average_iou = average_iou
            best_threshold = threshold

        if average_iou > 0.1:
            ths.append(threshold)
            ious.append(average_iou)

        print(f"th: {threshold} iou: {average_iou}")

    print(f"best th: {best_threshold}")

    return best_threshold, best_average_iou, ths, ious


def tune_score_threshold_based_on_hist(
    masks_gt, score_masks, target_fnr=0.05
) -> Tuple[dict, dict]:
    scores = np.array(score_masks).flatten()
    labels = np.array(masks_gt).astype(np.uint8).flatten()

    masks_gt = np.array(masks_gt).astype(np.bool_)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    eer_index = np.argmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_index]

    target_fnr_indx = np.argmin(np.abs(target_fnr - fnr))
    target_fnr_threshold = thresholds[target_fnr_indx]
    return {
        "th_err": eer_threshold,
        f"th_at_fnr{target_fnr*100:02}": target_fnr_threshold,
    }, {
        "fpr_at_eer": fpr[eer_index],
        "fnr_at_eer": fnr[eer_index],
        f"fpr_at_fpr{target_fnr*100:02}": fpr[target_fnr_indx],
    }
