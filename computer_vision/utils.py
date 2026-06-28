from tqdm import tqdm
from sklearn.cluster import KMeans
from typing import Any, List, Dict, Literal
from .metrics import intersection_over_union

import torch

def non_max_suppression(bboxes: List[List[float]], num_classes: int, iou_threshold: float, confidence_threshold: float) -> List[torch.Tensor]:
    """
    Non-Max-Supression method, sorts bounding boxes in function of the confidence and the overlay between the best box during the loop.
    
    :param List[List[float]] **bboxes**: Bounding boxes to sort, shape (batch, 6) = (batch, (id_class, confidence, box)) if num_classes > 1 otherwise shape (batch, 5) = (batch, (confidence, box)).
    :param int **num_classes**: Number of classes to predict.
    :param float **iou_threshold**: IoU threshold between two boxes to determine whether they are overlaid.
    :param float **confidence_threshold**: Threshold that bounding-box confidence scores must exceed.
    :return: Bounding boxes sorted by NMS.
    :rtype: List[List[float]]
    """
    scaling = 0 if num_classes > 1 else 1
    bboxes = sorted([box for box in bboxes if box[1-scaling] > confidence_threshold], key=lambda x: x[1-scaling], reverse=True)
    bouding_boxes = torch.tensor(bboxes)

    if len(bboxes) != 0:
        ious = intersection_over_union(bouding_boxes[..., 2-scaling], bouding_boxes[..., 2-scaling], is_aligned=False) < iou_threshold
        boolean_bboxes = [[] for _ in range(ious.shape[0])]

        for i in range(ious.shape[0] - 1):
            for j, intersection_bool in enumerate(ious[i][i + 1:].tolist()):
                boolean_bboxes[i + j + 1].append(intersection_bool)

        delete = 0
        for i in range(1, len(boolean_bboxes)):
            if torch.tensor(boolean_bboxes[i]).all() != True:
                bboxes.pop(i - delete)
                delete += 1
    return bboxes

def generate_bboxes_prior(annotations: Dict[str, Any], num_bboxes_prior_per_scale: int = 3, normalize: bool = False, box_format: Literal['xyxy', 'xywh', 'xcycwh'] = "xywh") -> torch.Tensor:
    """
    Method that generates bboxes prior for a given dataset.

    :param Dict[str, Any] **annotations**: Dictionary containing bounding boxes.
    :param int **num_bboxes_prior_per_scale**: Number of bboxes prior per scale. Set to `3`.
    :param bool **normalize**: Boolean that specifies if values must be normalized. Set to `False`.
    :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
    :return: Bounding Boxes prior.
    :rtype: Tensor
    """
    bboxes = []
    for (_, _bboxes), (_, sizes) in tqdm(zip(annotations["bboxes"].items(), annotations["sizes"].items()), total=len(annotations["bboxes"])):
        for bbox in _bboxes:
            bbox = bbox[1:]
            if normalize:
                bboxes.append([bbox[0] / sizes[0], bbox[1] / sizes[1], bbox[2] / sizes[0], bbox[3] / sizes[1]])
            else:
                bboxes.append(bbox)

    bboxes = torch.tensor(bboxes)

    if box_format == "xyxy":
        widths = bboxes[..., 2] - bboxes[..., 0]
        heights = bboxes[..., 3] - bboxes[..., 1]
    elif box_format == "xywh" or box_format == "xcycwh":
        widths = bboxes[..., 2]
        heights = bboxes[..., 3]

    whs = torch.stack((widths, heights), dim=1)
    kmeans = KMeans(n_clusters=3 * num_bboxes_prior_per_scale)
    kmeans.fit(whs)

    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    bboxes_prior = centers[torch.argsort(centers[..., 0] * centers[..., 1], descending=True)].reshape(3, num_bboxes_prior_per_scale, 2)
    return bboxes_prior