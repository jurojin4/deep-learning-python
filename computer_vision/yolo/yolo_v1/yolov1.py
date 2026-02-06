from math import floor
from typing import Literal

import os
import torch
import torch.nn as nn

# (convolutional neurons, kernel_size, stride, padding)
darknet_architecture = [
    (64, 7, 2, 3),
    "maxpooling",
    (192, 3, 1, 1),
    "maxpooling",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "maxpooling",
    [(256, 1, 1, 0), (512, 3, 1, 1)]*4,
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "maxpooling",
    [(512, 1, 1, 0), (1024, 3, 1, 1)]*2,
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1)]

class ConvBlock(nn.Module):
    """
    YOLOV1 Convolutional Block class.
    """
    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size: int, conv_stride: int, conv_padding: int, batch_normalization: bool = False, bias: bool = True):
        """
        Initializes the ConvBlock class.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param int **out_channels**: Number that represents the number of features maps, i.e the number of convolutional neurons doing a 2D convolution on the input.
        :param int **conv_kernel_size**: Size of the convolution kernel.
        :param int **conv_stride**: Size of the convolution stride.
        :param int **conv_padding**: Size of the convolution padding.
        :param bool **batch_normalization**: Set to `False`, if `True` then a batch normalization layer is added to the convolutional block.
        :param bool **bias**: Set to `True`, adds bias to the weighted sum before applying the activation function.
        """
        assert isinstance(out_channels, int), f"out_channels has to be an {int} instance, not {type(out_channels)}."
        assert isinstance(conv_kernel_size, int), f"conv_kernel_size has to be an {int} instance, not {type(conv_kernel_size)}."
        assert isinstance(conv_stride, int), f"conv_stride has to be an {int} instance, not {type(conv_stride)}."
        assert isinstance(conv_padding, int), f"conv_padding has to be an {int} instance, not {type(conv_padding)}."
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, bias=bias)

        self._has_batch_normalization = batch_normalization
        if batch_normalization:
            self.batch_normalization = nn.BatchNorm2d(out_channels)
        self.act_fn = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the ConvBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        if self._has_batch_normalization:
            return self.act_fn(self.batch_normalization(self.conv2d(x)))
        else:
            return self.act_fn(self.conv2d(x))

class Darknet(nn.Module):
    """
    Darknet class.\n\nBackbone of the YOLOV1 model.
    """
    def __init__(self, in_channels: int, img_size: int, batch_normalization: bool = False, bias: bool = True):
        """
        Initializes the Darknet class.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param int **img_size**: Size of the input image of dimension (Batch, Size, Size, 3).
        :param bool **batch_normalization**: Set to `False`, if `True` then a batch normalization layer is added to the convolutional block.
        :param bool **bias**: Set to `True`, adds bias to the weighted sum before applying the activation function.
        """
        super().__init__()
        self._img_size = img_size
        self._backbone = self._create_backbone(in_channels=in_channels, batch_normalization=batch_normalization, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the ConvBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return self._backbone(x)
        
    def _create_backbone(self, in_channels: int, batch_normalization: bool = False, bias: bool = True) -> nn.Sequential:
        """
        Method that creates the darknet network.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param bool **batch_normalization**: Set to `False`, if `True` then a batch normalization layer is added to the convolutional block.
        :param bool **bias**: Set to `True`, adds bias to the weighted sum before applying the activation function.
        :return: Darknet network.
        :rtype: Sequential
        """
        sequential = []
        for block in darknet_architecture:
            if isinstance(block, tuple):
                sequential.append(ConvBlock(in_channels=in_channels, out_channels=block[0], conv_kernel_size=block[1], conv_stride=block[2], conv_padding=block[3], batch_normalization=batch_normalization, bias=bias))
                self._img_size = self._size_image(self._img_size, kernel=block[1], stride=block[2], padding=block[3])
            
            if isinstance(block, str):
                if block == "maxpooling":
                    sequential.append(nn.MaxPool2d(2, 2))
                    self._img_size = self._size_image(self._img_size, kernel=2, stride=2, padding=0)
                    continue

            if isinstance(block, list):
                for b in block:
                    sequential.append(ConvBlock(in_channels=in_channels, out_channels=b[0], conv_kernel_size=b[1], conv_stride=b[2], conv_padding=b[3], batch_normalization=batch_normalization, bias=bias))
                    in_channels = b[0]
                    self._img_size = self._size_image(self._img_size, kernel=b[1], stride=b[2], padding=b[3])

                continue

            in_channels = block[0]
        return nn.Sequential(*sequential)
    
    def _size_image(self, size: int, kernel: int, stride: int, padding: int) -> int:
        """
        Method that calculates the 3D size image according height and width.

        :param int **kernel**: Size of the convolution kernel.
        :param int **stride**: Size of the convolution stride.
        :param int **padding**: Size of the convolution padding.
        :return: Image size after the convolution.
        :rtype: int
        """
        return floor((size + 2 * padding - kernel) / stride) + 1

class Head(nn.Module):
    """
    YOLOV1 Head class.
    """
    def __init__(self, size: int, C: int, S: int, B: int, mode: Literal["classification", "object_detection"]):
        """
        Initializes the Head class.

        :param int **size**: Image size after the forward darknet network.
        :param int **C**: Number of classes to predict.
        :param int **S**: Cells number along height and width.
        :param int **B**: Bounding boxes number in each cell.
        :param Literal["classification", "object_detection"] **mode**: String that defines the model mode.
        """
        super().__init__()
        self._S = S
        self._B = B
        self._C = C
        self._head = self._create_head(size=size, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the head.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return self._head(x)
    
    def _create_head(self, size: int, mode: Literal["classification", "object_detection"]) -> nn.Sequential:
        """
        :param int **size**: Image size after the forward darknet network.
        :param Literal["classification", "object_detection"] **mode**: String that defines the model mode.
        :return: Head network.
        :rtype: Sequential
        """
        if mode == "classification":
            return nn.Sequential(
                *[nn.Flatten(),
                nn.Linear(1024 * size * size, 4096),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, self._C),
                nn.Softmax(dim = -1)])
        else:
            return nn.Sequential(
                *[nn.Flatten(),
                nn.Linear(1024 * size * size, 4096),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, self._S * self._S * (self._C + (self._B * 5)))])

class YOLOV1(nn.Module):
    def __init__(self, in_channels: int, img_size: int, C: int, S: int = 7, B: int = 2, batch_normalization: bool = False, bias: bool = True, mode: Literal["classification", "object_detection"] = "object_detection"):
        """
        Initializes the YOLOV1 class.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param int **img_size**: Size of the input image of dimension (Batch, Size, Size, 3).
        :param int **C**: Number of classes to predict.
        :param int **S**: Cells number along height and width. Set to `7`.
        :param int **B**: Bounding boxes number in each cell. Set to `2`.
        :param bool **batch_normalization**: Set to `False`, if `True` then a batch normalization layer is added to the convolutional block.
        :param bool **bias**: Set to `True`, adds bias to the weighted sum before applying the activation function.
        :param Literal["classification", "object_detection"] **mode**: String that defines the model mode. Set to `object_detection`.
        """
        assert isinstance(in_channels, int), f"in_channels has to be an {int} instance, not {type(in_channels)}."
        assert isinstance(img_size, int), f"img_size has to be an {int} instance, not {type(img_size)}."
        assert isinstance(batch_normalization, bool), f"batch_normalization has to be a {bool} instance, not {type(batch_normalization)}."
        assert isinstance(bias, bool), f"bias has to be a {bool} instance, not {type(bias)}."
        assert isinstance(mode, str) and mode in ["classification", "object_detection"], f"mode has to be a {str} instance and in the given list:\n[\"classification\", \"object_detection\"]"
        super().__init__()
        self._backbone = Darknet(in_channels=in_channels, img_size=img_size, batch_normalization=batch_normalization, bias=bias)

        if mode == "classification":
            self.S = None
            self.B = None
        else:
            self.S = S
            self.B = B
        self.C = C

        self._head = Head(self._backbone._img_size, C=C, S=self.S, B=self.B, mode=mode)
        self._mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the YOLOV1 model.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        assert isinstance(x, torch.Tensor), f"x has to a {torch.Tensor} instance, not {type(x)}."
        return self._head(self._backbone(x))
    
    def save(self, dirname: str, filename: str) -> None:
        """
        Method that saves the actual model state.
        
        :param str **dirname**: Dirname path.
        :param str **filename**: Name of the saved checkpoint without extension.
        """
        assert isinstance(dirname, str), f"dirname has to be a {str} instance, not {type(dirname)}."
        assert isinstance(filename, str), f"filename has to be a {str} instance, not {type(filename)}."
        if filename.find(".") != -1:
            print("filename should not have extension.")
            filename = filename.split(".")[0]
        state_dict = self.state_dict()
        checkpoint_path = os.path.join(dirname, filename + ".pth")
        torch.save(state_dict, checkpoint_path)

    def load(self, weights_path: str, all: bool = False) -> None:
        """
        Method that loads weights from a file (.pth).
        
        :param str **weights_path**: Path of the saved weights.
        :param bool **all**: Boolean that allows loading either all weights or only the convolution weights. Set to `False`.
        """
        if all:
            self.load_state_dict(torch.load(weights_path))
            print("Model load completely.")
        else:
            model_weights = torch.load(weights_path)
            for key in list(model_weights.keys()):
                if "head" in key:
                    model_weights.pop(key)
            self.load_state_dict(model_weights, strict=False)
            print("Model load partially.")


def init_weights(m):
    if isinstance(m, ConvBlock):
        nn.init.uniform(m.conv2d.weight, a=0, b=1)
        m.conv2d.weight.data *= 1 # scale down to ~1e-2
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-1, b=1)
        m.weight.data *= 100

