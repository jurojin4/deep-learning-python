from collections import Counter
from typing import Dict, List, Tuple, Union
from yolo_tools import intersection_over_union

import torch
import torch.nn as nn

class Metric(nn.Module):
    """
    Base metric class.
    """
    name: str
    def __init__(self):
        """
        Initiliazes the Metric class.
        """
        super().__init__()

    def _prepare_gts_and_preds(self, ground_truths: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method that prepares ground truths and predictions to have same shape for loss calculation.

        :param Tensor **ground_truths**: Tensor containing the values to predict.
        :param Tensor **predictions**: Tensor containing the model predictions.

        :return: Ground Truths and Predictions with same shape.
        :rtype: Tensor
        """
        if ground_truths.shape == predictions.shape:
            return ground_truths, predictions
        
        if len(ground_truths.shape) < len(predictions.shape):
            ground_truths = ground_truths.unsqueeze(-1)

        if ground_truths.shape[-1] == predictions.shape[-1]:
            if predictions.dtype == bool:
                return ground_truths, predictions.type(int)
            else:
                return ground_truths, predictions
        else:
            preds = predictions.argmax(axis=-1, keepdims=True)

            shape = ground_truths.shape
            new_predictions = torch.zeros(shape)

            for i in range(ground_truths.shape[0]):
                new_predictions[i][int(preds[i].item())] = 1

            return ground_truths, new_predictions
    
class Accuracy(Metric):
    """
    Accuracy metric class.
    """
    def __init__(self):
        """
        Initializes the Accuracy class.
        """
        super().__init__()
        super().__setattr__("name", "accuracy")

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Method that calculates accuracy between predictions and ground_truths.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Accuracy, value between 0 to 1.
        :rtype: float
        """
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        n = y.shape[0]
        return (y == y_pred).sum().item() / n

class Precision(Metric):
    """
    Precision metric class.
    """
    def __init__(self, num_classes: int, detailed: bool = False):
        """
        Initializes the Precision class.

        :param int **num_classes**: Number of classes.
        :param bool **detailed**: Boolean that adds precision per class. Set to `False.
        """
        super().__init__()
        super().__setattr__("name", "precision")
        self.num_classes = num_classes
        self.detailed = detailed

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor) -> Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]:
        """
        Method that calculates precision between predictions and ground_truths. if detailed is `True` then precision per class is added.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Average precision (value between 0 to 1) and if detailed is `True` precision per class.
        :rtype: Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]
        """ 
        if self.detailed:
            return self._forward_d(y, y_pred)
        else:
            return self._forward_nd(y, y_pred)

    def _forward_nd(self, y: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Method that calculates precision between predictions and ground_truths.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Average precision (value between 0 to 1) and precision per class.
        :rtype: float
        """ 
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        classes_present = y.unique()

        average_precision = 0

        for class_label in classes_present:
            true_positive = ((y_pred == class_label.item()) & (y == class_label.item())).sum().item()
            predicted_positive = (y_pred == class_label.item()).sum().item()

            average_precision += true_positive / predicted_positive if predicted_positive != 0 else 0
        
        average_precision /= len(classes_present)
        return average_precision
    
    def _forward_d(self, y: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, Dict[int, Union[float, None]]]:
        """
        Method that calculates precision between predictions and ground_truths in a detailed way.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Average precision (value between 0 to 1) and precision per class.
        :rtype: Tuple[float, Dict[int, Union[float, None]]]
        """
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        classes_present = y.unique()

        precision_per_class = dict([(int(label.item()), None) for label in y.unique()])

        for class_label in classes_present:
            true_positive = ((y_pred == class_label.item()) & (y == class_label.item())).sum().item()
            predicted_positive = (y_pred == class_label.item()).sum().item()

            precision = true_positive / predicted_positive if predicted_positive != 0 else 0
            precision_per_class[int(class_label)] = precision
        
        average_precision = sum([item for _, item in precision_per_class.items() if item is not None]) / len(classes_present)
        return average_precision, precision_per_class

class Recall(Metric):
    """
    Recall metric class.
    """
    def __init__(self, num_classes: int, detailed: bool = False):
        """
        Initializes the Recall class.

        :param int **num_classes**: Number of classes.
        :param bool **detailed**: Boolean that adds precision per class. Set to `False.
        """
        super().__init__()
        super().__setattr__("name", "recall")
        self.num_classes = num_classes
        self.detailed = detailed

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor) -> Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]:
        """
        Method that calculates recall between predictions and ground_truths. if detailed is `True` then recall per class is added.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Average recall (value between 0 to 1) and if detailed is `True` precision per class.
        :rtype: Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]
        """ 
        if self.detailed:
            return self._forward_d(y, y_pred)
        else:
            return self._forward_nd(y, y_pred)

    def _forward_nd(self, y: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Method that calculates recall between predictions and ground_truths.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Average recall (value between 0 to 1).
        :rtype: float
        """
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)
        
        classes_present = y.unique()

        recall = 0.

        for class_label in classes_present:
            true_positive = ((y_pred == class_label.item()) & (y == class_label.item())).sum().item()
            actual_positive = (y == class_label.item()).sum().item()
            
            recall += true_positive / actual_positive if actual_positive != 0 else 0

        recall /= len(classes_present)
        return recall

    def _forward_d(self, y: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, Dict[int, Union[float, None]]]:
        """
        Method that calculates recall between predictions and ground_truths in a detailed way.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Average recall (value between 0 to 1) and recall per class.
        :rtype: Tuple[float, Dict[int, Union[float, None]]]
        """
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        recall_per_class = dict([(int(label.item()), None) for label in y.unique()])

        classes_present = y.unique()

        for class_label in classes_present:
            true_positive = ((y_pred == class_label.item()) & (y == class_label.item())).sum().item()
            actual_positive = (y == class_label.item()).sum().item()
            recall = true_positive / actual_positive if actual_positive != 0 else 0
            recall_per_class[int(class_label)] = recall

        average_recall = sum([item for _, item in recall_per_class.items() if item is not None]) / len(classes_present)
        return average_recall, recall_per_class

class F1_Score(Metric):
    """
    F1-Score metric class.
    """
    def __init__(self, num_classes: int, detailed: bool = False):
        """
        Initializes the F1_score class.

        :param int **num_classes**: Number of classes.
        :param bool **detailed**: Boolean that adds precision per class. Set to `False.
        """
        super().__init__()
        super().__setattr__("name", "f1_score")
        self.num_classes = num_classes
        self.precision = Precision(num_classes, detailed=detailed)
        self.recall = Recall(num_classes, detailed=detailed)
        self.detailed = detailed

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor) -> Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]:
        """
        Method that calculates f1-score between predictions and ground_truths. if detailed is `True` then f1-score per class is added.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Average f1-score (value between 0 to 1) and if detailed is `True` precision per class.
        :rtype: Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]
        """ 
        if self.detailed:
            return self._forward_d(y, y_pred)
        else:
            return self._forward_nd(y, y_pred)

    def _forward_nd(self, y: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Method that calculates f1-score between predictions and ground_truths.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Average F1-score (value between 0 to 1).
        :rtype: float
        """
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        avg_prec = self.precision(y, y_pred)
        avg_rec = self.recall(y, y_pred)  

        average_f1 = 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec) if avg_prec != 0 and avg_rec != 0 else 0

        return average_f1
        
    def _forward_d(self, y: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, Dict[int, Union[float, None]]]:
        """
        Method that calculates f1-score between predictions and ground_truths in a detailed way.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Average F1-score (value between 0 to 1) and F1-score per class.
        :rtype: Tuple[float, Dict[int, Union[float, None]]]
        """
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        avg_prec, prec_per_class = self.precision(y, y_pred)
        avg_rec, rec_per_class = self.recall(y, y_pred)
        average_f1 = 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec) if avg_prec != 0 and avg_rec != 0 else 0

        f1_per_class = {}
        for (key, prec), (_, rec) in zip(prec_per_class.items(), rec_per_class.items()):
            f1_per_class[key] = (2 * (prec * rec) / (prec + rec)) if (prec + rec) != 0 else 0

        return average_f1, f1_per_class

class Mean_Average_Precision(Metric):
    """
    Mean Average Precision metric class.
    """
    def __init__(self, num_classes: int, batch_size: int, iou_threshold: float = 0.5, detailed: bool = False):
        """
        Initializes Mean Average Precision class.

        :param int **num_classes**: Number of classes.
        :param int **batch**: Size of the batch.
        :param float **iou_threshold**: Threshold for IoU. Set to `0.5`.
        :param bool **detailed**: Boolean that adds precision per class. Set to `False.
        """
        super().__init__()
        super().__setattr__("name", f"mAP@{int(iou_threshold * 100)}")
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._iou_threshold = iou_threshold
        self.detailed = detailed

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor) -> Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]:
        """
        Method that calculates mAP between predictions and ground_truths. if detailed is `True` then f1-score per class is added.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Average mAP (value between 0 to 1) and if detailed is `True` mAP per class.
        :rtype: Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]
        """ 
        if self.detailed:
            return self._forward_d(y, y_pred)
        else:
            return self._forward_nd(y, y_pred)

    def _forward_nd(self, predictions_bboxes: List[List[Union[float, int]]], ground_truths_bboxes: List[List[Union[float, int]]]) -> float:
        """
        Method that calculates mAP between predictions and ground_truths.

        :param List[List[Union[float, int]]] **predictions_bboxes**: Bounding boxes of predictions.
        :param List[List[Union[float, int]]] **ground_truths_bboxes**: Bounding boxes of ground_truths.
        :return: Average mAP (value between 0 to 1).
        :rtype: float
        """
        mean_average_precision = []

        for preds_bboxes, gts_bboxes in zip(predictions_bboxes, ground_truths_bboxes):
            total_bboxes_per_batch = [0] * self._batch_size
            for gt_bbox in gts_bboxes:
                total_bboxes_per_batch[int(gt_bbox[0].item())] += 1

            amount_bboxes = Counter([gt[0] for gt in gts_bboxes])
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)
                            
            preds_bboxes.sort(key=lambda box: box[2], reverse=True)
            true_positive = torch.zeros((len(preds_bboxes)))
            false_positive = torch.zeros((len(preds_bboxes)))
            total_gt_bboxes = len(gts_bboxes)     

            if total_gt_bboxes == 0:
                continue

            for pred_idx, pred_bbox in enumerate(preds_bboxes):
                best_iou = 0

                for gt_idx, gt in enumerate(gts_bboxes):
                    iou = intersection_over_union(pred_bbox[3:], gt[3:])

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                    if best_iou > self._iou_threshold:
                        if amount_bboxes[best_gt_idx] == 0:
                            true_positive[pred_idx] = 1
                            amount_bboxes[best_gt_idx] = 1
                        else:
                            false_positive[pred_idx] = 1
                    else:
                        false_positive[pred_idx] = 1

                true_positive_cumsum = true_positive.cumsum(dim=0)
                false_positive_cumsum = false_positive.cumsum(dim=0)

                recalls = true_positive_cumsum / (total_gt_bboxes + 1e-9)
                precisions = true_positive_cumsum / (true_positive_cumsum + false_positive_cumsum + 1e-9)

                mean_average_precision.append(torch.trapz(precisions, recalls).item())

        if len(mean_average_precision) != 0:
            mean_average_precision = sum(mean_average_precision) / len(mean_average_precision)
        else:
            mean_average_precision = 0.

        return mean_average_precision

    def _forward_d(self, predictions_bboxes: List[List[Union[float, int]]], ground_truths_bboxes: List[List[Union[float, int]]]) -> Tuple[float, Dict[int, Union[float, None]]]:
        """
        Method that calculates mAP between predictions and ground_truths.

        :param List[List[Union[float, int]]] **predictions_bboxes**: Bounding boxes of predictions.
        :param List[List[Union[float, int]]] **ground_truths_bboxes**: Bounding boxes of ground_truths.
        :return: Average mAP (value between 0 to 1) and mAP per class.
        :rtype: Tuple[float, Dict[int, Union[float, None]]]
        """
        average_precision_per_class = {}

        for label, (preds_bboxes, gts_bboxes) in enumerate(zip(predictions_bboxes, ground_truths_bboxes)):
            total_bboxes_per_batch = [0] * self._batch_size
            for gt_bbox in gts_bboxes:
                total_bboxes_per_batch[int(gt_bbox[0].item())] += 1

            amount_bboxes = Counter([gt[0] for gt in gts_bboxes])
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)
                            
            preds_bboxes.sort(key=lambda box: box[2], reverse=True)
            true_positive = torch.zeros((len(preds_bboxes)))
            false_positive = torch.zeros((len(preds_bboxes)))
            total_gt_bboxes = len(gts_bboxes)     

            if total_gt_bboxes == 0:
                continue

            for pred_idx, pred_bbox in enumerate(preds_bboxes):
                best_iou = 0

                for gt_idx, gt in enumerate(gts_bboxes):
                    iou = intersection_over_union(pred_bbox[3:], gt[3:])

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                    if best_iou > self._iou_threshold:
                        if amount_bboxes[best_gt_idx] == 0:
                            true_positive[pred_idx] = 1
                            amount_bboxes[best_gt_idx] = 1
                        else:
                            false_positive[pred_idx] = 1
                    else:
                        false_positive[pred_idx] = 1

                true_positive_cumsum = true_positive.cumsum(dim=0)
                false_positive_cumsum = false_positive.cumsum(dim=0)

                recalls = true_positive_cumsum / (total_gt_bboxes + 1e-9)
                precisions = true_positive_cumsum / (true_positive_cumsum + false_positive_cumsum + 1e-9)

                if label in average_precision_per_class.keys():
                    average_precision_per_class[label].append(torch.trapz(precisions, recalls).item())
                else:
                    average_precision_per_class[label] = [torch.trapz(precisions, recalls).item()]
            
            if label in average_precision_per_class.keys():
                if average_precision_per_class[label] != 0:
                    average_precision_per_class[label] = sum(average_precision_per_class[label]) / len(average_precision_per_class[label])
                else:
                    average_precision_per_class[label] = 0

        if len(average_precision_per_class) != 0:
            mean_average_precision = sum([item for _, item in average_precision_per_class.items()]) / len(average_precision_per_class)
        else:
            mean_average_precision = 0.

        return mean_average_precision, average_precision_per_class