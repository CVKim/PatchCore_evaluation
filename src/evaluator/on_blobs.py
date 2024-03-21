import cv2
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Union


def extract_blobs_binary(binary_mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Extract connected components from a binary image mask.

    Args:
        binary_mask (np.ndarray): A binary image mask where regions to be labeled are white (1)
                                  and the background is black (0). Must be a 2D array of type np.uint8.

    Returns:
        Tuple[int, np.ndarray]: A tuple containing the number of unique connected components (excluding background)
                                and an array with the same size as `binary_mask` where each pixel's label is indicated.
    """
    binary_mask = binary_mask.astype(np.uint8)
    num_labels_with_bg, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    num_labels = num_labels_with_bg - 1
    return num_labels, labels


def extract_blobs(mask: np.ndarray, num_classes: int) -> Tuple[int, np.ndarray, dict]:
    """
    Extracts and classifies blobs from a mask for multiple classes.

    Args:
        mask (np.ndarray): 2D array with each pixel's value indicating its class (shape: [height, width]).
                           0 is background, and positive integer corresponds to an object class.
        num_classes (int): Number of distinct classes in the mask.

    Returns:
        Tuple[int, np.ndarray, dict]:
        - num_labels (int): Total number of unique blobs across all classes.
        - labels (np.ndarray): 2D array with unique labels for each blob (shape: [height, width]).
        - label2class (dict): Maps blob labels to their respective class.

    Processes each class in the mask separately to identify and label blobs. Labels are unique
    across different classes.
    """
    num_labels = 0
    label2class = {}
    for c in range(1, num_classes + 1):
        binary_mask = mask == c
        num_blob_labels, blob_labels = extract_blobs_binary(binary_mask)

        for i in range(1, num_blob_labels + 1):
            label2class[i + num_labels] = c

        if c == 1:
            labels = blob_labels
        else:
            blob_labels[blob_labels != 0] += num_labels
            # assert np.sum(labels[blob_labels != 0]) == 0
            labels += blob_labels

        num_labels += num_blob_labels

    return num_labels, labels, label2class


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Computes the Intersection over Union (IoU) of two masks.

    Args:
        mask1 (np.ndarray): First binary mask (2D array).
        mask2 (np.ndarray): Second binary mask (2D array).

    Returns:
        float: IoU score between mask1 and mask2. Returns 0 if the union is empty.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def compute_average_iou(masks1, masks2):
    total_iou = 0
    count = 0
    for mask1, mask2 in zip(masks1, masks2):
        # Compute IoU
        iou = compute_iou(mask1, mask2)
        total_iou += iou
        count += 1

    average_iou = total_iou / count if count != 0 else 0
    return average_iou


def _compute_confusion_matrix(
    mask_gt: np.ndarray, mask_pred: np.ndarray, num_classes: int, iou_threshold: float
) -> np.ndarray:
    """
    Computes a confusion matrix for a single pair of ground truth and predicted masks.

    Args:
        mask_gt (np.ndarray): Ground truth mask (2D array of integers). Each integer corresponds to a class (0 to background).
        mask_pred (np.ndarray): Predicted mask (2D array of integers). Each integer corresponds to a class (0 to background).
        num_classes (int): Number of classes excluding the background.
        iou_threshold (float): Threshold for IoU to consider a prediction as a true positive. It must be between 0 and 1.

    Returns:
        np.ndarray: Confusion matrix (2D array of shape [num_classes + 1, num_classes + 1]).
    """

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    num_bloblabels_gt, blob_labels_gt, label2class_gt = extract_blobs(
        mask_gt, num_classes
    )
    num_bloblabels_pred, blob_labels_pred, label2class_pred = extract_blobs(
        mask_pred, num_classes
    )

    # todo: complete the below
    predlabel2gtlabel = defaultdict(list)
    for i in range(1, num_bloblabels_gt + 1):
        max_iou = 0
        matching_pred_class = 0
        for j in range(1, num_bloblabels_pred + 1):
            iou = compute_iou(
                blob_labels_gt == i, blob_labels_pred == j
            )  # iou between blob_pred and blob_gt
            if iou >= iou_threshold:
                predlabel2gtlabel[j].append((i, iou))
                if iou > max_iou:
                    matching_pred_class = label2class_pred[j]
                    max_iou = max(max_iou, iou)

        confusion_matrix[label2class_gt[i], matching_pred_class] += 1

    for j in range(1, num_bloblabels_pred + 1):
        if len(predlabel2gtlabel[j]) == 0:
            confusion_matrix[0, label2class_pred[j]] += 1

    if compute_iou(blob_labels_pred == 0, blob_labels_gt == 0) >= iou_threshold:
        confusion_matrix[0, 0] += 1

    return confusion_matrix


def compute_confusion_matrix(
    masks_gt: Union[List[np.ndarray], np.ndarray],
    masks_pred: Union[List[np.ndarray], np.ndarray],
    num_classes: int,
    iou_threshold: float,
) -> np.ndarray:
    """
    Computes the confusion matrix for a set of ground truth and predicted masks.

    Args:
        masks_gt (Union[List[np.ndarray], np.ndarray]): Ground truth masks. Can be a list of 2D arrays
                                                        or a 3D array with shape (num_masks, height, width).
                                                        Each element should be a natural number (including 0),
                                                        where 0 indicates the background and positive integers
                                                        indicate object classes.
        masks_pred (Union[List[np.ndarray], np.ndarray]): Predicted masks, in the same format as masks_gt.
        num_classes (int): Number of classes, excluding the background.
        iou_threshold (float): IoU threshold for considering a prediction as a true positive.

    Returns:
        np.ndarray: Aggregate confusion matrix. A 2D array of shape [num_classes + 1, num_classes + 1],
                    where rows represent ground truth classes and columns represent predicted classes.

    The function computes a confusion matrix for each pair of corresponding ground truth and predicted masks.
    It sums these matrices to obtain an aggregate confusion matrix. This matrix is useful for evaluating the
    classification performance, with each cell [i, j] indicating the count of samples of ground truth class i
    predicted as class j.
    """

    assert len(np.unique(masks_gt)) <= num_classes + 1
    assert len(np.unique(masks_pred)) <= num_classes + 1

    masks_gt, masks_pred = np.array(masks_gt), np.array(masks_pred)

    assert masks_gt.shape == masks_pred.shape

    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    for mask_gt, mask_pred in zip(masks_gt, masks_pred):
        confusion_matrix += _compute_confusion_matrix(
            mask_gt, mask_pred, num_classes, iou_threshold
        )
    return confusion_matrix


def compute_metrics(
    masks_gt: np.ndarray, score_masks: np.ndarray, score_threshold: float, iou_threhold: float=0.5,
) -> Tuple[int, int, int, int]:
    """
    Computes TP, TN, FP, FN from ground truth and predicted score masks.

    Args:
        masks_gt (np.ndarray): Ground truth masks with shape (N, H, W), where N is the number of images,
                               H and W are height and width of the masks. The values are class labels.
        score_masks (np.ndarray): Predicted score masks, either shape (N, H, W, C) for multiple classes
                                  or (N, H, W) for a single class. C is the number of classes including
                                  the background.
        score_threshold (float): Threshold to convert score masks to binary masks.

    Returns:
        Tuple[int, int, int, int]: A tuple of (TP, TN, FP, FN).

    If score_masks is (N, H, W, C), each pixel's class is determined by the highest score among object
    classes if it exceeds the threshold, else it's the background. If score_masks is (N, H, W),
    it's binarized using the score threshold.
    """
    assert len(masks_gt) == len(score_masks)
    if score_masks.ndim == 3:
        num_classes = 1
        masks_pred = (score_masks >= score_threshold).astype(np.uint8)
    elif score_masks.ndim == 4:
        num_classes = score_masks.shape[-1] - 1
        _masks_pred = np.argmax(score_masks, axis=-1, keepdims=True)
        _max_scores = np.take_along_axis(score_masks, _masks_pred, axis=-1).squeeze(-1)
        _masks_pred = _masks_pred.squeeze(-1)
        assert _masks_pred.shape == _max_scores.shape
        _masks_pred[_max_scores < score_threshold] = 0
        masks_pred = _masks_pred
    else:
        raise ValueError()

    assert masks_gt.shape == masks_pred.shape

    confusion_matrix = compute_confusion_matrix(
        masks_gt, masks_pred, num_classes, iou_threshold=iou_threhold
    )

    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1:].sum()
    FN = confusion_matrix[1:, 0].sum()
    TP = np.diag(confusion_matrix)[1:].sum()

    return TP, TN, FP, FN


if __name__ == "__main__":
    np.random.seed(5)

    batch_size = 10
    dim = 20
    num_classes = 1
    masks_pred = np.random.randint(
        0, num_classes + 1, (batch_size, dim, dim), dtype=np.uint8
    )
    masks_gt = np.random.randint(
        0, num_classes + 1, (batch_size, dim, dim), dtype=np.uint8
    )
    iou_threshold = 0.01

    confusion_matrix = compute_confusion_matrix(
        masks_gt, masks_pred, num_classes, iou_threshold
    )

    print(f"confusion matrix: {confusion_matrix}")

    # score_masks = np.random.rand(batch_size, dim, dim)
    score_masks = masks_pred
    score_threshold = 0.0
    
    # TP, TN, FP, FN = compute_metrics(
    #     masks_gt, score_masks, score_threshold, iou_threshold
    # )
    # print(f"TP: {TP} TN: {TN} FP: {FP} FN: {FN}")
    _score_masks = np.concatenate([np.zeros_like(score_masks[:, :, :, None]), score_masks[:, :, :, None]], axis=-1)

    _TP, _TN, _FP, _FN = compute_metrics(
        masks_gt, _score_masks, score_threshold, iou_threshold
    )
    
    print(f"_TP: {_TP} _TN: {_TN} _FP: {_FP} _FN: {_FN}")


def remove_small_blobs(binary_masks, min_size):

    binary_masks = binary_masks.astype(np.uint8)

    processed_binary_masks = [_remove_small_blobs(binary_mask, min_size) for binary_mask in binary_masks]
    return processed_binary_masks


def _remove_small_blobs(binary_mask: np.ndarray, min_size):
    assert isinstance(binary_mask, np.ndarray)
    assert binary_mask.ndim == 2

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)

    # Filter out small blobs
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if width < min_size and height < min_size:
            binary_mask[labels == label] = 0

    return binary_mask