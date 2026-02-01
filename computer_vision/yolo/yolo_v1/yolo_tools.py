from typing import List, Literal, Tuple, Union

import torch

eps = 1e-9

def intersection_over_union(bboxes_preds: torch.Tensor, bboxes_gts: torch.Tensor, box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh") -> torch.Tensor:
    """
    Method that calculates IoU between bounding boxes of predictions and ground_truths.

    :param Tensor **bboxes_preds**: Predictions bounding boxes of shapes (B, 4).
    :param Tensor **bboxes_gts**: Ground Truths bounding boxes of shapes (B, 4).
    :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
    :return: Intersection over Union of shapes (B, 1).
    :rtype: Tensor
    """
    if box_format == "xyxy":
        preds_x1 = bboxes_preds[..., 0:1]
        preds_y1 = bboxes_preds[..., 1:2]
        preds_x2 = bboxes_preds[..., 2:3]
        preds_y2 = bboxes_preds[..., 3:4]

        ground_truths_x1 = bboxes_gts[..., 0:1]
        ground_truths_y1 = bboxes_gts[..., 1:2]
        ground_truths_x2 = bboxes_gts[..., 2:3]
        ground_truths_y2 = bboxes_gts[..., 3:4]

    elif box_format == "xywh":
        preds_x1 = bboxes_preds[..., 0:1]
        preds_y1 = bboxes_preds[..., 1:2]
        preds_x2 = bboxes_preds[..., 0:1] + bboxes_preds[..., 2:3]
        preds_y2 = bboxes_preds[..., 1:2] + bboxes_preds[..., 3:4]

        ground_truths_x1 = bboxes_gts[..., 0:1]
        ground_truths_y1 = bboxes_gts[..., 1:2]
        ground_truths_x2 = bboxes_gts[..., 0:1] + bboxes_gts[..., 2:3]
        ground_truths_y2 = bboxes_gts[..., 1:2] + bboxes_gts[..., 3:4]
    elif box_format == "xcycwh":
        preds_x1 = bboxes_preds[..., 0:1] - (bboxes_preds[..., 2:3] / 2)
        preds_y1 = bboxes_preds[..., 1:2] - (bboxes_preds[..., 3:4] / 2)
        preds_x2 = bboxes_preds[..., 2:3] + preds_x1
        preds_y2 = bboxes_preds[..., 3:4] + preds_y1

        ground_truths_x1 = bboxes_gts[..., 0:1] - (bboxes_gts[..., 2:3] / 2)
        ground_truths_y1 = bboxes_gts[..., 1:2] - (bboxes_gts[..., 3:4] / 2)
        ground_truths_x2 = bboxes_gts[..., 2:3] + ground_truths_x1
        ground_truths_y2 = bboxes_gts[..., 3:4] + ground_truths_y1

    x1 = torch.max(preds_x1, ground_truths_x1)
    y1 = torch.max(preds_y1, ground_truths_y1)
    x2 = torch.min(preds_x2, ground_truths_x2)
    y2 = torch.min(preds_y2, ground_truths_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    preds_area = abs((preds_x2 - preds_x1) * (preds_y2 - preds_y1))
    ground_truths_area = abs((ground_truths_x2 - ground_truths_x1) * (ground_truths_y2 - ground_truths_y1))

    return intersection / (preds_area + ground_truths_area - intersection + eps)

def non_max_suppression(bboxes: List[torch.Tensor], iou_threshold: float, confidence_threshold: float, alternative: bool = False) -> List[torch.Tensor]:
    """
    Non-Max-Supression method, sorts bounding boxes in function of the confidence and the overlay between the best box during the loop.
    
    :param List[Tensor] **bboxes**: Bounding boxes to sort, shape (batch, 6).
    :param float **iou_threshold**: IoU threshold between two boxes to determine whether they are overlaid.
    :param float **confidence_threshold**: Threshold that bounding-box confidence scores must exceed.
    :param bool **alternative**: Boolean that adds a condition during NMS (label difference).
    :return: Bounding boxes sorted by NMS.
    :rtype: List[Tensor]
    """
    bboxes = sorted([box for box in bboxes if box[1] > confidence_threshold], key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        best_box = bboxes.pop(0)

        if alternative:
            bboxes = [box for box in bboxes if intersection_over_union(best_box[2:], box[2:]) < iou_threshold or best_box[0] != box[0]]
        else:
            bboxes = [box for box in bboxes if intersection_over_union(best_box[2:], box[2:]) < iou_threshold]

        bboxes_after_nms.append(best_box)
    return bboxes_after_nms

def get_bboxes(batch: int, S: int, B: int, C: int, predictions: torch.Tensor, ground_truths: torch.Tensor, iou_threshold: float, confidence_threshold: float, alternative: bool = False) -> Tuple[List[List[Union[float, int]]], List[List[Union[float, int]]]]:
    """
    Method that converts and sorts bounding boxes with NMS method.

    :param int **batch**: Number of samples in ground_truths and predictions from the dataset.
    :param int **S**: Cells number along height and width.
    :param int **B**: Bounding boxes number in each cell.
    :param int **C**: Number of classes to predict.
    :param Tensor **predictions**: Predictions bounding boxes of shapes (batch, S, S, C + 6 * B).
    :param Tensor **ground_truths**: Ground Truths bounding boxes of shapes (batch, S, S, C + 6 * B).
    :param float **iou_threshold**: IoU threshold between two boxes to determine whether they are overlaid.
    :param float **confidence_threshold**: Threshold that bounding-box confidence scores must exceed.
    :param bool **alternative**: Boolean that adds a condition during NMS (label difference).
    :return: Bounding boxes converted and sorted.
    :rtype: Tuple[List[List[Union[float, int]]], List[List[Union[float, int]]]]
    """
    gts_bboxes = cellboxes_to_boxes(ground_truths, S, B, C)
    preds_bboxes = cellboxes_to_boxes(predictions, S, B, C)

    all_preds = [[] for _ in range(C)]
    all_ground_truths = [[] for _ in range(C)]
    for idx in range(batch):
        nms_boxes = non_max_suppression(preds_bboxes[idx], iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, alternative=alternative)
        for nms_box in nms_boxes:
            all_preds[int(nms_box[0].item())].append(torch.cat((torch.tensor([idx]), nms_box)))

        for box in gts_bboxes[idx]:
            if box[1] == 1:
                all_ground_truths[int(box[0].item())].append(torch.cat((torch.tensor([idx]), box)))
    
    del gts_bboxes
    del preds_bboxes
    
    return all_preds, all_ground_truths

def get_bboxes_preds(batch: int, S: int, B: int, C: int, predictions: torch.Tensor, iou_threshold: float, confidence_threshold: float, alternative: bool = False) -> List[List[Union[float, int]]]:
    """
    Method that converts and sorts bounding boxes with NMS method.

    :param int **batch**: Number of samples in predictions from the dataset.
    :param int **S**: Cells number along height and width.
    :param int **B**: Bounding boxes number in each cell.
    :param int **C**: Number of classes to predict.
    :param Tensor **predictions**: Predictions bounding boxes of shapes (batch, S, S, C + 6 * B).
    :param float **iou_threshold**: IoU threshold between two boxes to determine whether they are overlaid.
    :param float **confidence_threshold**: Threshold that bounding-box confidence scores must exceed.
    :param bool **alternative**: Boolean that adds a condition during NMS (label difference).
    :return: Bounding boxes converted and sorted.
    :rtype: List[List[Union[float, int]]]
    """
    preds_bboxes = cellboxes_to_boxes(predictions, S, B, C)

    all_preds = [[] for _ in range(C)]
    for idx in range(batch):
        nms_boxes = non_max_suppression(preds_bboxes[idx], iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, alternative=alternative)
        for nms_box in nms_boxes:
            all_preds[int(nms_box[0].item())].append(torch.cat((torch.tensor([idx]), nms_box)))

    del preds_bboxes
    
    return all_preds

def cellboxes_to_boxes(bboxes: torch.Tensor, S: int, B: int, C: int) -> List[torch.Tensor]:
    """
    Method that converts bounding boxes from cell to global representation.

    :param Tensor **bboxes**: .
    :param int **S**: Cells number along height and width.
    :param int **B**: Bounding boxes number in each cell.
    :param int **C**: Number of classes to predict.
    :return: Bounding boxes converted.
    :rtype: List[torch.Tensor]
    """
    bboxes = bboxes.to("cpu")
    batch_size = bboxes.shape[0]

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)

    boundingboxes = []
    for i in range(C, C + 5 * B, 5):
        label = bboxes[..., :C].argmax(-1).unsqueeze(-1)
        score = bboxes[..., i].unsqueeze(-1)
        x = (1 / S) * (bboxes[..., i+1:i+2] + cell_indices)
        y = (1 / S) * (bboxes[..., i + 2:i + 3] + cell_indices.permute(0, 2, 1, 3))
        w = (1 / S) * bboxes[..., i + 3:i + 4]
        h = (1 / S) * bboxes[..., i + 4:i + 5]

        converted_bboxes = torch.cat((x, y, w, h), dim=-1)
        converted_bboxes = torch.cat((label, score, converted_bboxes), dim=-1)

        boundingboxes.append(converted_bboxes)

    boundingboxes = torch.cat(boundingboxes, dim=-1)
    boundingboxes = boundingboxes.reshape((boundingboxes.shape[0], S * S * B , 6))

    all_bboxes = []
    for batch_idx in range(boundingboxes.shape[0]):
        all_bboxes.append(boundingboxes[batch_idx])
    return all_bboxes