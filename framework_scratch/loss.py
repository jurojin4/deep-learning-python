from typing import Literal, Tuple
from module import Module

import tensyx as ts

class Loss(Module):
    """
    Base loss function class .
    """
    def __init__(self):
        """
        Initializes the Loss class.
        """
        super().__init__(has_parameters=False)

    def _prepare_gts_and_preds(self, ground_truths: ts.Tensor, predictions: ts.Tensor) -> Tuple[ts.Tensor, ts.Tensor]:
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
            ground_truths = ground_truths.expand_dims(-1)

        if ground_truths.shape[-1] == predictions.shape[-1]:
            if predictions.dtype == bool:
                return ground_truths, predictions.astype(int)
            else:
                return ground_truths, predictions
        else:
            preds = predictions.argmax(axis=-1, keepdims=True)

            shape = ground_truths.shape
            new_predictions = ts.zeros(shape)

            for i in range(ground_truths.shape[0]):
                new_predictions[i][int(preds[i].item())] = 1

            return ground_truths, new_predictions

class CrossEntropyLoss(Loss):
    """
    CrossEntropyLoss class.
    """
    def __init__(self, reduction: Literal["mean", "sum"] = "mean"):
        """
        Initializes the CrossEntropyLoss class.

        :param Literal["mean", "sum"], optional **reduction**: Set to `mean`, method used to reduct the loss output.
        """
        super().__init__()
        self.reduction = reduction
        self.forward = self._forward
        self._define_name(name=self.__class__.__name__, reduction=reduction)

    def _forward(self, y: ts.Tensor, y_pred: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the loss function the truth and the prediction.

        :param Tensor **y**: Truth tensor.
        :param Tensor **y_pred**: Prediction tensor.
        :return: Loss value tensor.
        :rtype: Tensor
        """
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        logits = y_pred.logits()
        loss = - (y * (ts.log((ts.exp(logits - logits.max(axis=-1, keepdims=True)) / ts.exp(logits - logits.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)).clip(1e-7, 1-1e-7))))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

class LogLoss(Loss):
    """
    LogLoss class.
    """
    def __init__(self, reduction: Literal["mean", "sum"] = "mean"):
        """
        Initializes the LogLoss class.

        :param Literal["mean", "sum"], optional **reduction**: Set to `mean`, method used to reduct the loss output.
        """
        super().__init__()
        self.reduction = reduction
        self.forward = self._forward
        self._define_name(name=self.__class__.__name__, reduction=reduction)

    def _forward(self, y: ts.Tensor, y_pred: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the loss function the truth and the prediction.

        :param Tensor **y**: Truth tensor.
        :param Tensor **y_pred**: Prediction tensor.
        :return: Loss value tensor.
        :rtype: Tensor
        """
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        loss = (- y * ts.log(y_pred) + (y - 1) * ts.log(1 - y_pred))
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()

class MSELoss(Loss):
    """
    Mean Squared Error Loss class.
    """
    def __init__(self, reduction: Literal["mean", "sum"] = "mean"):
        """
        Initializes the MSELoss class.

        :param Literal["mean", "sum"], optional **reduction**: Set to `mean`, method used to reduct the loss output.
        """
        super().__init__()
        self.reduction = reduction
        self.forward = self._forward
        self._define_name(name=self.__class__.__name__, reduction=reduction)

    def _forward(self, y: ts.Tensor, y_pred: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the loss function the truth and the prediction.

        :param Tensor **y**: Truth tensor.
        :param Tensor **y_pred**: Prediction tensor.
        :return: Loss value tensor.
        :rtype: Tensor
        """
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        loss = ts.square(y_pred - y)
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()