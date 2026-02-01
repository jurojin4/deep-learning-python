from typing import Any, Callable, Union, Tuple

import tensyx as ts

def _measure_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(f"Metric [{type(self).__name__}] is missing the required \"measure\" function.")

class Metric:
    """
    Base metric class.
    """
    name: str
    def __init__(self):
        """
        Initiliazes the Metric class.
        """
        self.measure: Callable[..., Any] = _measure_unimplemented

    def __call__(self, *args, **kwargs):
        """
        Built-in Python method that allows to call an instance like function.
        """
        return self.measure(*args, **kwargs)

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
        self.measure = self._measure

    def _measure(self, y: ts.Tensor, y_pred: ts.Tensor) -> float:
        """
        Method that measures predictions compared to ground_truths.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Value between 0 to 1.
        :rtype: float
        """
        assert(y.shape == y_pred.shape), f"The predictions has to have the same size than ground truths:\n- y.shape = {y.shape}\n- y_pred.shape = {y_pred.shape}"
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        n = y.shape[0]
        return (y == y_pred).sum().item() / n

class Precision(Metric):
    def __init__(self, num_classes: int):
        """
        Initializes the Precision class.

        :param num_classes: Number of classes.
        """
        super().__init__()
        super().__setattr__("name", "precision")
        self.num_classes = num_classes
        self.measure = self._measure

    def _measure(self, y: ts.Tensor, y_pred: ts.Tensor) -> float:
        """
        Method that measures predictions compared to ground_truths.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Value between 0 to 1.
        :rtype: float
        """
        assert(y.shape == y_pred.shape), f"The predictions has to have the same size than ground truths:\n- y.shape = {y.shape}\n- y_pred.shape = {y_pred.shape}"
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        precision_total = 0.
        classes_present = y.unique(type=int)

        for class_label in classes_present:
            true_positive = ((y_pred == class_label) & (y == class_label)).sum().item()
            predicted_positive = (y_pred == class_label).sum().item()

            precision = true_positive / predicted_positive if predicted_positive != 0 else 0
            precision_total += precision
        
        average_precision = precision_total / len(classes_present)
        return average_precision

class Recall(Metric):
    def __init__(self, num_classes: int):
        """
        Initializes the Recall class.

        :param num_classes: Number of classes.
        """
        super().__init__()
        super().__setattr__("name", "recall")
        self.num_classes = num_classes
        self.measure = self._measure

    def _measure(self, y: ts.Tensor, y_pred: ts.Tensor) -> float:
        """
        Method that measures predictions compared to ground_truths.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Value between 0 to 1.
        :rtype: float
        """
        assert(y.shape == y_pred.shape), f"The predictions has to have the same size than ground truths:\n- y.shape = {y.shape}\n- y_pred.shape = {y_pred.shape}"
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        recall_total = 0.

        classes_present = y.unique(type=int)

        for class_label in classes_present:
            true_positive = ((y_pred == class_label) & (y == class_label)).sum().item()
            actual_positive = (y == class_label).sum().item()
            recall = true_positive / actual_positive if actual_positive != 0 else 0
            recall_total += recall

        average_recall = recall_total / len(classes_present)
        return average_recall

class F1_Score(Metric):
    def __init__(self, num_classes: int):
        """
        Initializes the F1_score class.

        :param num_classes: Number of classes.
        """
        super().__init__()
        super().__setattr__("name", "f1_score")
        self.num_classes = num_classes
        self.precision = Precision(num_classes)
        self.recall = Recall(num_classes)
        self.measure = self._measure

    def _measure(self, y: ts.Tensor, y_pred: ts.Tensor) -> float:
        """
        Method that measures predictions compared to ground_truths.

        :param Tensor **y**: Ground Truths.
        :param Tensor **y_pred**: Model predictions.
        :return: Value between 0 to 1.
        :rtype: float
        """
        assert(y.shape == y_pred.shape), f"The predictions has to have the same size than ground truths:\n- y.shape = {y.shape}\n- y_pred.shape = {y_pred.shape}"
        y, y_pred = self._prepare_gts_and_preds(y, y_pred)

        prec = self.precision(y, y_pred)
        rec = self.recall(y, y_pred)
        average_f1 = 2 * (prec * rec) / (prec + rec) if prec != 0 and rec != 0 else 0

        return average_f1