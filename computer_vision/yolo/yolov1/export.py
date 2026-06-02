from typing import List
from .yolov1 import YOLOV1
from .yolo_tools import get_bboxes_preds
from ...export import ONNXExport, get_args_parser

import torch

class YOLOV1Export(ONNXExport):
    """
    YOLOV1 ONNX export class.
    """
    def __init__(self, weights_path: str):
        """
        Initializes the YOLOV1Export class.

        :param str **weights_path**: Path of the weights.
        """
        self.model_box_format = "xywh"
        super().__init__(weights_path)

    def _define_model(self, weights_path: str) -> None:
        """
        Property that defines the model.

        :param str **weights_path**: Path of the weights.
        """
        self._model = YOLOV1(in_channels=3, image_height=self._logs["image_size"][0], image_width=self._logs["image_size"][1], num_classes=self._logs["num_classes"], mode=self._logs["dataset_type"])
        self._model.load(weights_path, all=True)

    def _model_tools(self, predictions: torch.Tensor) -> List[List[List[float]]]:
        """
        Methid that defines model's tools.

        :param Tensor **predictions**: Model's predictions.
        :return: Bounding boxes converted and sorted.
        :rtype: List[List[List[float]]]
        """
        return get_bboxes_preds(predictions, self._model.S, self._model.B, num_classes=self._num_classes, iou_threshold=self._iou_threshold_overlap, confidence_threshold=self._confidence_threshold)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    export = YOLOV1Export(weights_path=args.weights_path)
    export(images_directory=args.images_directory)