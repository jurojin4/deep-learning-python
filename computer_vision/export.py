from PIL import Image
from time import time
from typing import Any, List, Union
from torchvision.transforms import ToTensor

import os
import torch
import pickle
import argparse
import onnxruntime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ONNXExport:
    """
    Base class for ONNX export
    """
    def __init__(self, weights_path: str):
        """
        Initializes the ONNXExport class.

        :param str **weights_path**: Path of the weights.
        """
        self.weights_path = weights_path
        self._define_logs(os.path.join(os.path.dirname(weights_path), "logs.pickle"))
        self._define_model(weights_path)

    def _define_model(self, weights_path: str):
        raise NotImplementedError(f'ONNXExport [{type(self).__name__}] is missing the required "_define_model" method.')
    
    def _define_logs(self, logs_path: str) -> None:
        """
        Initializes the ONNXExport class.

        :param str **logs_path**: Path of the logs.
        """
        with open(logs_path, "rb") as file:
            self._logs = pickle.load(file)

        self._num_classes = self._logs["num_classes"]
        self._iou_threshold_overlap = self._logs["iou_threshold_overlap"]
        self._confidence_threshold = self._logs["confidence_threshold"]
        self._image_height, self._image_width = self._logs["image_size"]

    def __call__(self, images_directory: Union[str, None] = None) -> None:
        """
        Built-in Python method that allows to call an instance like function.

        :param Union[str, None] **images_directory**: Directory path of the images. Set to `None`. 
        """
        self._model.eval()
        height, width = self._logs["image_size"]
        example_inputs = torch.randn(1, 3, height, width)
        with torch.inference_mode():
            torch.onnx.export(
                self._model,                     
                example_inputs,                        
                os.path.join(os.path.dirname(self.weights_path), "model_checkpoint.onnx"),
                opset_version=18,
                dynamo=False,                
                training=torch.onnx.TrainingMode.EVAL,
                do_constant_folding=True,
                keep_initializers_as_inputs=False,
                input_names=["input"],
                output_names=["predictions"])
            
        self._ort_session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(self.weights_path), "model_checkpoint.onnx"), providers=["CUDAExecutionProvider"])
            
        self._testing_images(height, width, images_directory)

    def _testing_images(self, width: int, height: int, images_directory: str) -> None:
        """
        Method that tests torch and onnx models on images.

        :param int **height**: Height of the image.
        :param int **width**: Width of the image.
        :param str **images_directory**: Path of the directory that contains images to test.
        """
        plt.ion()
        _, ax = plt.subplots(1, 1)
        if images_directory is not None:
            totensor = ToTensor()
            for filename in sorted(os.listdir(images_directory)):
                ax.clear()
                image = Image.open(os.path.join(images_directory, filename), 'r')
                image_size = image.size

                inputs = totensor(image.resize((width, height))).unsqueeze(0)
                input = [tensor.unsqueeze(0).numpy(force=True) for tensor in inputs]
                input = {input_arg.name: input_value for input_arg, input_value in zip(self._ort_session.get_inputs(), input)}

                t0 = time()
                model_predictions = self._model(inputs)
                t1 = time()
                onnx_predictions = self._ort_session.run(None, input)[0]
                t2 = time()

                diff = torch.abs(model_predictions - torch.from_numpy(onnx_predictions))

                print(f"Max in absolute: {diff.max()}")
                print(f"Mean: {diff.mean()}")

                print(f"PyTorch Model inference time: {(t1-t0):.4f} second(s)")
                print(f"ONNX model inference time: {(t2-t1):.4f} second(s)\n")

                self._draw(torch.from_numpy(onnx_predictions), image_size[1], image_size[0], ax, color="red")
                self._draw(model_predictions, image_size[1], image_size[0], ax, color="blue")
                ax.imshow(image)
                plt.pause(1e-15)
                plt.show()

    def _draw(self, predictions: torch.Tensor, height: int, width: int, ax, color: str):
        """
        Method that draws bounding boxes on image.

        :param Tensor **predictions**: Predictions of a model.
        :param int **height**: Height of the image.
        :param int **width**: Width of the image.
        :param str **color**: Color of the bboxes.
        """
        predictions = self._model_tools(predictions)
        for label, preds in enumerate(predictions):
            for bbox in preds:
                batch, conf = bbox[:2]
                bbox = self._bbox_transformation(bbox[2:])
                rect = patches.Rectangle((bbox[0] * width, bbox[1] * height), bbox[2] * width, bbox[3] * height, linewidth=2, edgecolor=color, facecolor='none', alpha=0.5)
                ax.add_patch(rect)
        print("\n")

    def _model_tools(self, predictions: Any):
        raise NotImplementedError(f'ONNXExport [{type(self).__name__}] is missing the required "_model_tools" method.')
    
    def _bbox_transformation(self, bbox: List[float]) -> List[float]:
        """
        Method that transforms bounding boxes to a define box format.

        :param List[float] **bbox**: Bounding boxes.
        :return: Bounding boxes with a specified format.
        :rtype: List[float]
        """
        if self.model_box_format == "xyxy":
            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        elif self.model_box_format == "xywh":
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
        else:
            x = bbox[0] - (bbox[2] / 2)
            y = bbox[1] - (bbox[3] / 2)
            w = bbox[2]
            h = bbox[3]
        
        return [x, y, w, h]

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='ONNXExport', add_help=add_help)
    parser.add_argument("--weights_path", default=None, type=str, help='Path of the weights')
    parser.add_argument("--images_directory", default=None, type=str, help='Path of the images')

    return parser

