from tqdm import tqdm
from typing import Dict, List, Literal, Tuple, Union
from metrics import Accuracy, Precision, Recall, F1_Score
from loss import Loss
from module import Module, Sequential
from data import DataLoader, Data_Metrics
from optimizer import Optimizer
from animation.gif import AGIF

import nn
import tensyx as ts

class Trainer:
    """
    Class that train models
    """
    def __init__(self, epochs: int, model: Union[Module, Sequential], optimizer: Optimizer, loss: Loss, dataset_type: str, train_loader: DataLoader, validation_loader: Union[DataLoader, None]= None, num_classes: Union[int, None] = None, metrics: Union[List[str], str, None] = None, verbose: bool = True, gif: bool = False):
        """
        Initializes the Trainer class.

        :param int **epochs**: Number of iterations during the training.
        :param Union[Module, Sequential] **model**: Model to train.
        :param Optimizer **optimizer**: Algorithm used to update the weights.
        :param str **dataset_type**: Type of the dataset.
        :param Loss **loss**: Function/Metric to minimize during the training.
        :pram DataLoader **train_loader**: DataLoader for "train" part.
        :pram DataLoader **validation_loader**: DataLoader for "validation" part. Set to `None`.
        :param int **num_classes: Number of class in the dataset.
        :param Union[List[str], str, None] **metrics**: Metrics used during the training.
        :param bool **verbose**: . Set to `True`.
        :param bool **gif**: If set `True` then a gif of the training is created.
        """
        # :param bool **animation**: Set to `False`, if `True` then a window is displayed to see the evolution of the model during the training according metrics.

        super().__init__()
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.dataset_type = dataset_type
        self.loss = loss
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self._prediction_condition = self._defines_prediction_condition

        self.metrics = {Accuracy.__name__.lower(): Accuracy(), Precision.__name__.lower(): Precision(num_classes), Recall.__name__.lower(): Recall(num_classes), F1_Score.__name__.lower(): F1_Score(num_classes)}
        self._metrics_to_use = []

        if isinstance(metrics, list):
            if len(metrics) != 0:
                for metric in metrics:
                        try:
                            self._metrics_to_use.append(self.metrics.pop(metric))
                        except:
                            print(f"{metric} does not belong in the available metrics.")
                            continue                             
        elif isinstance(metrics, str):
            try:
                self._metrics_to_use.append(self.metrics.pop(metrics))
            except:
                print(f"{metric} does not belong in the available metrics.")

        metrics_for_data = ["loss"] + metrics
        self.data_metrics = Data_Metrics(stages=("Train", train_loader.dataset), metrics=metrics_for_data)
        if validation_loader is not None:
            self.data_metrics.add_stages(stages=("Validation", validation_loader.dataset))

        self._verbose = verbose
        self._gif = gif
        
    @property
    def _defines_prediction_condition(self) -> Union[float, None]:
        """
        Property that defines if necessary the condition to apply to predictions during the training.
        """
        last_module = self.model[-1]
        if isinstance(last_module, nn.Sigmoid):
            return 1 / 2
        elif isinstance(last_module, (nn.Identity, nn.TanH)):
            return 0.
        elif isinstance(last_module, (nn.Softmax)):
            return None
        else:
            return 0

    def __call__(self) -> Data_Metrics:
        """
        Built-in Python method that allows to call an instance like function, launch the training.

        :return: Metrics and parameters of each epoch during the training.
        :rtype: Data_Metrics.
        """
        for epoch in range(0, self.epochs + 1):
            if self._verbose:
                print(f"Epoch: {epoch}/{self.epochs}")
            train_metrics = self._train(epoch)
            if self._verbose:
                self._display_metrics(train_metrics, "Train")
            self.data_metrics.update(stage="Train", metrics=train_metrics)

            if self.validation_loader is not None:
                validation_metrics = self._validation()
                if self._verbose:
                    self._display_metrics(validation_metrics, "Validation")
                self.data_metrics.update(stage="Validation", metrics=validation_metrics)

            self.data_metrics.model_parameters.append(self.model.get_parameters)

            if self._verbose:
                print("\n")
        if self._gif:
            agif = AGIF(self.model, self.data_metrics, self.dataset_type)
            agif()
        return self.data_metrics

    def _display_metrics(self, metrics: Dict[str, float], mode: Literal["Train", "Validation"]) -> None:
        """
        Method that displays metrics into the terminal.

        :param Dict[str, float] **metrics**: Dictionary containing as keys metrics name and as items theirs values.
        :param Literal["Train", "Validation"] **mode**: String that can be equal to `Train` or `Validation`.
        """         
        for metric_name, measure in metrics.items():
            print(f"{mode} {metric_name}: {round(measure, 3)}", end=" | ")
        print("\n")

    def _train(self, epoch: int) -> Dict[str, float]:
        """
        Method that trains the model.

        :param int **epoch**: Current training epoch.
        :return: Dictionary containing the average metrics for the epoch.
        :rtype: Dict[str, float]
        """
        metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
        metrics["loss"] = []

        if self._verbose:
            loop = tqdm(self.train_loader, leave=True)
        else:
            loop = self.train_loader

        for x, y in loop:
            y_pred = self.model(x)

            loss = self.loss(y, y_pred)
            metrics["loss"].append(loss.item())

            if epoch != 0:
                loss.backward()
                self.optimizer.step(epoch)

            if self._defines_prediction_condition is None:
                y, y_pred = self._prepare_gts_and_preds(y, y_pred)
            else:
                y, y_pred = self._prepare_gts_and_preds(y, y_pred > self._prediction_condition)

            if y.shape[-1] != 1:
                y, y_pred = y.argmax(axis=-1), y_pred.argmax(axis=-1)

            for metric in self._metrics_to_use:
                metrics[metric.name].append(metric(y, y_pred))

            if self._verbose:
                loop.set_postfix([("Train_" + name, values[-1]) for name, values in metrics.items()])

        self._average(metrics)
        return metrics
        
    def _validation(self) -> Dict[str, float]:
        """
        Method that evaluates the model.

        :return: Dictionary containing the average metrics for the epoch.
        :rtype: Dict[str, float]
        """
        metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
        metrics["loss"] = []

        if self._verbose:
            loop = tqdm(self.validation_loader, leave=True)
        else:
            loop = self.validation_loader

        for x, y in loop:
            y_pred = self.model(x)
            loss = self.loss(y, y_pred)
            metrics["loss"].append(loss.item())

            if self._defines_prediction_condition is None:
                y, y_pred = self._prepare_gts_and_preds(y, y_pred)
            else:
                y, y_pred = self._prepare_gts_and_preds(y, y_pred > self._prediction_condition)

            if y.shape[-1] != 1:
                y, y_pred = y.argmax(axis=-1), y_pred.argmax(axis=-1)

            for metric in self._metrics_to_use:
                metrics[metric.name].append(metric(y, y_pred))

            if self._verbose:
                loop.set_postfix([("Validation_" + name, values[-1]) for name, values in metrics.items()])

        self._average(metrics)
        return metrics
    
    def _average(self, metrics: Dict[str, List[float]]) -> None:
        """
        Method that averages the metrics.

        :param Dict[str, List[float]] **metrics**: Metrics used during the training.
        """
        for metric_name, values in metrics.items():
            metrics[metric_name] = sum(values) / len(values)

    def _prepare_gts_and_preds(self, ground_truths: ts.Tensor, predictions: ts.Tensor) -> Tuple[ts.Tensor, ts.Tensor]:
        """
        Method that prepares the ground truths and the predictions in order to have same size for the measurements.

        :param ts.Tensor **ground_truths**: Tensor that contains ground truths of the given batch.
        :param ts.Tensor **predictions**: Tensor that contains model predictions of the given batch.

        :return: Ground Truths and model predictions for the given batch.
        :rtype: Tuple[ts.Tensor, ts.Tensor]
        """
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