from tqdm import tqdm
from yolov1 import YOLOV1
from typing import Literal
from loss import YOLOV1Loss
from torch.optim import Optimizer
from yolo_tools import get_bboxes
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Dict, List, Literal, Tuple, Union
from metrics import Accuracy, Precision, Recall, F1_Score, Mean_Average_Precision
from prepare_dataset import prepare_cifar10, prepare_vocdetection, prepare_tiny_imagenet, Compose, YOLOV1Dataset

import os
import time
import torch
import random
import pickle
import torch.nn as nn
import torch.optim as optim

def train_model(dataset_name: Literal["cifar10", "pascalvoc2007", "pascalvoc2012", "tiny-imagenet200"], dataset_path: str, epochs: int, size: int, batch_size: int, learning_rate: float = 1e-5, milestones: Union[List[int], None] = None, detailed: bool = False, save: bool = False, save_metric: Literal["accuracy", "loss", "mAP", "precision", "recall", "f1_score"] = "loss", delete: bool = False, weights_path: Union[str, None] = None, load_all: bool = False, experiment_name: Union[str, None] = None) -> None:
    """
    Method that trains the model following chosen arguments.

    :param Literal["cifar10", "pascalvoc2007", "pascalvoc2012", "tiny-imagenet200"] **dataset_name**: Name of the dataset.
    :param str **dataset_path**: Path of the dataset, where it will be stored.
    :param int **epochs**: Number of iterations during the training.
    :param int **size**: Image size.
    :param int **batch_size**: Batch size.
    :param float **learning_rate**: Coefficient to apply during gradient descent. Set to `1e-5`.
    :param Union[List[int], None] **milestones**: Integers list of epochs. Set to `None`.
    :param bool **detailed**: Boolean that allows to have metrics for each class. Set to `False`.
    :param bool **save**: Boolean that saves metrics. Set to `False`
    :param Literal["accuracy", "loss", "mAP", "precision", "recall", "f1_score"] **save_metric**: Saves the model according the given metric. Set to `loss`.
    :param bool **delete**: Boolean that allows to delete the downloaded dataset. Set to `False`.
    :param Union[str, None] **weights_path: Path of the weights. Set to `None`.
    :param bool **load_all**: Boolean that allows to load partially or completely the model. Set to `False`
    :param Union[str, None] **experiment_name**: Name of the experiment. Set to `None`.
    """
    if dataset_name == "cifar10":
        train, validation, weights, num_classes, categories, dataset_name, mode = prepare_cifar10(path=dataset_path, delete=delete)
    elif dataset_name == "tiny-imagenet200":
        train, validation, weights, num_classes, categories, dataset_name, mode = prepare_tiny_imagenet(path=dataset_path, delete=delete)
    elif dataset_name == "pascalvoc2007":
        train, validation, weights, num_classes, categories, dataset_name, mode = prepare_vocdetection(path=dataset_path, year="2007")
    else:
        train, validation, weights, num_classes, categories, dataset_name, mode = prepare_vocdetection(path=dataset_path, year="2012")

    model = YOLOV1(in_channels=3, img_size=size, C=num_classes, batch_normalization=True, mode=mode)

    if weights_path is not None:
        model.load(weights_path=weights_path, all=load_all)

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

    if mode == "classification":
        loss = nn.CrossEntropyLoss(weight=torch.tensor(weights).to("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        loss = YOLOV1Loss(model.C, model.S, model.B)

    if milestones is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = None
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones)


    compose = Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean=0, std=255)])

    train_dataset = YOLOV1Dataset(dataset=train, C=model.C, S=model.S, B=model.B, mode=mode, compose=compose)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=os.cpu_count(),
                              pin_memory=False,
                              drop_last=True)
    
    validation_dataset = YOLOV1Dataset(dataset=validation, C=model.C, S=model.S, B=model.B, mode=mode, compose=compose)
    validation_loader = DataLoader(dataset=validation_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=os.cpu_count(),
                              pin_memory=False,
                              drop_last=True)
    
    if experiment_name is None:
        experiment_name = "".join([chr(random.randint(97, 122)) for _ in range(5)] + [str(random.randint(0, 1000))])

    trainer = Trainer(epochs=epochs,
            model=model, 
            categories=categories, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            loss=loss, 
            train_loader=train_loader,
            validation_loader=validation_loader,
            mode=mode,
            detailed=detailed,
            save=save,
            save_metric=save_metric,
            dataset_name=dataset_name,
            experiment_name=experiment_name)
    
    trainer()


class Trainer:
    """
    Class that train models.
    """
    def __init__(self, epochs: int, model: Union[nn.Module, nn.Sequential], categories: Dict[str, int], optimizer: Optimizer, scheduler, loss, train_loader: DataLoader, validation_loader: Union[DataLoader, None]= None, mode: Literal["classification", "object_detection"] = "object_detection", verbose: bool = True, detailed: bool = False, save: bool = False, save_metric: Union[str, None] = None, dataset_name: Union[str, None] = None, experiment_name: Union[str, None] = None):
        """
        Initializes the Trainer class.

        :param int **epochs**: Number of iterations during the training.
        :param Union[Module, Sequential] **model**: Model to train.
        :param optim **optimizer**: Algorithm used to update the weights.
        :param **scheduler**: Object that manages the learning rate during the training.
        :param **loss**: Function/Metric to minimize during the training.
        :pram DataLoader **train_loader**: DataLoader for "train" part.
        :pram DataLoader **validation_loader**: DataLoader for "validation" part. Set to `None`.
        :param int **categories**: Number of class in the dataset.
        :param Literal["classification", "object_detection"] **mode**: String that defines the training mode. Set to `object_detection`.
        :param Union[List[str], str, None] **metrics**: Metrics used during the training.
        :param bool **verbose**: Set to `True`.
        :param bool **detailed**: Boolean that adds precision per class. Set to `False.
        :param bool **save**: Boolean that allows to save the model during the training according accuracy metric value. Set to `False`.
        :param str **dataset_name**: Name of the dataset used. Set to `None`.
        """
        assert isinstance(epochs, int), f"epochs has to be a {int} instance, not {type(epochs)}."
        assert isinstance(model, (nn.Module, nn.Sequential)), f"model has to be a {nn.Module} or {nn.Sequential} instance, not {type(model)}."
        assert isinstance(categories, dict), f"categories has to be a {dict} instance, not {type(categories)}."
        assert isinstance(optimizer, Optimizer), f"optimizer has to be a {Optimizer} instance, not {type(optimizer)}."
        assert isinstance(train_loader, DataLoader), f"train_loader has to be a {DataLoader} instance, not {type(train_loader)}."
        assert isinstance(validation_loader, DataLoader) or validation_loader == None, f"validation_loader has to be a {DataLoader} instance (or None), not {type(validation_loader)}."
        assert isinstance(mode, str), f"mode has to be a {str} instance, not {type(mode)}."
        assert isinstance(verbose, bool), f"verbose has to be a {bool} instance, not {type(verbose)}."
        assert isinstance(save, bool), f"save has to be a {bool} instance, not {type(save)}."
        assert isinstance(save_metric, str) or save_metric == None, f"save_metric has to be a {str} instance (or None), not {type(save_metric)}."
        assert isinstance(dataset_name, str) or dataset_name == None, f"dataset_name has to be a {str} instance (or None), not {type(dataset_name)}."
        super().__init__()
        self._epochs = epochs
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loss = loss
        self._categories = categories
        self._num_classes = len(categories)
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        self._mode = mode

        if mode == "classification":
            self._metrics = {Accuracy.__name__.lower(): Accuracy(), Precision.__name__.lower(): Precision(self._num_classes, detailed), Recall.__name__.lower(): Recall(self._num_classes, detailed), F1_Score.__name__.lower(): F1_Score(self._num_classes, detailed)}
        else:
            boolean = True
            while boolean:
                try:
                    num_threshold = int(input("Choose how many IoU threshold you want it to measure the model (strictly Postive Integer only): "))
                    assert isinstance(num_threshold, int) and num_threshold >= 1, f"Only stricly positive integer not {num_threshold}."

                    self._iou_threshold_overlay = float(input("IoU threshold between two boxes to determine whether they are overlaid. (between 0. and 1.): "))
                    assert isinstance(self._iou_threshold_overlay, float) and self._iou_threshold_overlay >= 0. and self._iou_threshold_overlay <= 1., f"IoU threshold (overlay) has to be between 0. and 1., not {self._iou_threshold_overlay}."                    

                    self._confidence_threshold = float(input("Confidence threshold: "))
                    assert isinstance(self._confidence_threshold, float) and self._confidence_threshold >= 0. and self._confidence_threshold <= 1., f"Confidence threshold (overlay) has to be between 0. and 1., not {self._confidence_threshold}"
        
                    if num_threshold == 1:
                        print(f"Default IoU threshold: 0.5 (50%)")
                        mAP = Mean_Average_Precision(num_classes=self._num_classes, batch_size=train_loader.batch_size, detailed=detailed)
                        self._metrics = {mAP.name: mAP}
                        boolean = False
                    else:
                        self._metrics = {}
                        for i in range(num_threshold):
                            _boolean = True
                            while _boolean:
                                try:
                                    iou = float(input(f"IoU threshold {i+1} (between 0. and 1.): "))
                                    assert iou < 1 and iou > 0., "IoU threshold should be 0.<= IoU threshold <= 1."
                                    mAP = Mean_Average_Precision(num_classes=self._num_classes, batch_size=train_loader.batch_size, iou_threshold=iou, detailed=detailed)
                                    self._metrics[mAP.name] = mAP
                                    _boolean = False
                                except Exception as e:
                                    print(e)
                                    continue

                        boolean = False
                except Exception as e:
                    print(e)
                    continue

        self._metrics_to_use = [item for _, item in self._metrics.items()]

        self._verbose = verbose
        self._detailed = detailed
        self._save = save

        if save:
            dirname = os.path.dirname(__file__)
            if not os.path.exists(os.path.join(dirname, "model_saves", dataset_name)):
                os.mkdir(os.path.join(dirname, "model_saves", dataset_name))
            
            localtime = time.localtime()
            self._save_path = os.path.join(dirname, "model_saves", dataset_name, "_".join([self._model.__class__.__name__, mode, str(localtime.tm_year), str(localtime.tm_yday), str(localtime.tm_hour), str(localtime.tm_min), str(localtime.tm_sec)]))
            os.mkdir(self._save_path)
            self._save_log_path = os.path.join(self._save_path, "checkpoint_log.txt")
            
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break
            
            image_size = train_loader.dataset[0][0].size()[1:]
            model_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
            with open(self._save_log_path, "a") as file:
                file.write(f"experiment name:{experiment_name}|dataset type:{mode}|dataset:{dataset_name}|classes:{self._num_classes}|image size:{image_size[0], image_size[1]}|model parameters:{model_parameters}|epochs:{epochs}|batch:{train_loader.batch_size}|optimizer:{optimizer.__class__.__name__}|learning rate: {lr}|loss:{loss.__class__.__name__}")
                if scheduler is not None:
                    file.write(f"|scheduler:{scheduler.__class__.__name__}|milestones:{scheduler.milestones}")
                if mode == "object_detection":
                    file.write(f"|iou threshold overlay:{self._iou_threshold_overlay}|confidence threshold:{self._confidence_threshold}")

            with open(os.path.join(self._save_path, f"categories.pickle"), 'wb') as handle:
                pickle.dump(self._categories, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self._save_metric = save_metric
        if self._save_metric is not None:
            if self._save_metric not in list(self._metrics.keys()) + ["loss"]:
                self._save_metric = "loss"
        else:
            self._save_metric = "accuracy"

        self._device_model = next(model.parameters()).device

        if self._mode == "classification":
            self._fp = 3
        else:
            self._fp = 5

    def __call__(self) -> Union[nn.Module, nn.Sequential]:
        """
        Built-in Python method that allows to call an instance like function, launch the training.

        :return: Model trained.
        :rtype: Union[nn.Module, nn.Sequential]
        """
        train_metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
        train_metrics["loss"] = []
        train_metrics_per_class = {}

        if self._validation_loader is not None:
            validation_metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
            validation_metrics["loss"] = []
            validation_metrics_per_class = {}

        if self._save_metric == "loss":
            best_metric = torch.inf
        else:
            best_metric = 0

        for epoch in range(0, self._epochs + 1):
            if self._verbose:
                print(f"Epoch: {epoch}/{self._epochs}")
            
            if self._detailed:
                _train_metrics, _train_metrics_per_class = self._train(epoch)
            else:
                _train_metrics = self._train(epoch)

            for key in _train_metrics:
                train_metrics[key].append(_train_metrics[key])

            if self._detailed:
                for key in _train_metrics_per_class:
                    if key not in train_metrics_per_class:
                        train_metrics_per_class[key] = dict([(label, []) for label in range(self._num_classes)])
                    for label, value in _train_metrics_per_class[key].items():
                        train_metrics_per_class[key][label].append(value)

            if self._verbose:
                self._display_metrics(_train_metrics, "Train")

            if self._validation_loader is not None:
                if self._detailed:
                    _validation_metrics, _validation_metrics_per_class = self._validation()
                else:
                    _validation_metrics = self._validation()

                for key in _validation_metrics:
                    validation_metrics[key].append(_validation_metrics[key])

                if self._detailed:
                    for key in _validation_metrics_per_class:
                        if key not in validation_metrics_per_class:
                            validation_metrics_per_class[key] = dict([(label, []) for label in range(self._num_classes)])
                        for label, value in _validation_metrics_per_class[key].items():
                            validation_metrics_per_class[key][label].append(value)

                if self._verbose:
                    self._display_metrics(_validation_metrics, "Validation")

            if self._scheduler is not None:
                self._scheduler.step()

            if self._verbose:
                print("\n")

            if self._save and epoch != 0:
                if self._validation_loader is not None:
                    if self._save_metric == "loss":
                        if best_metric > _validation_metrics[self._save_metric]:
                            self._save_model(epoch, _validation_metrics[self._save_metric], self._save_metric, "validation")
                            best_metric = _validation_metrics[self._save_metric]
                    else:
                        if best_metric < _validation_metrics[self._save_metric]:
                            self._save_model(epoch, _validation_metrics[self._save_metric], self._save_metric, "validation")
                            best_metric = _validation_metrics[self._save_metric]
                else:
                    if self._save_metric == "loss":
                        if best_metric > _train_metrics[self._save_metric]:
                            self._save_model(epoch, _train_metrics[self._save_metric], self._save_metric, "train")
                            best_metric = _train_metrics[self._save_metric]
                    else:
                        if best_metric < _train_metrics[self._save_metric]:
                            self._save_model(epoch, _train_metrics[self._save_metric], self._save_metric, "train")
                            best_metric = _train_metrics[self._save_metric]

            if self._save:
                self._save_metrics(metrics=train_metrics, stage="train")
                self._save_metrics(metrics=train_metrics_per_class, stage="train", name="pc")

                if self._validation_loader is not None:
                    self._save_metrics(metrics=validation_metrics, stage="validation")
                    self._save_metrics(metrics=validation_metrics_per_class, stage="validation", name="pc")

        self._save_model(epoch, None, None, None)

        return self._model

    def _display_metrics(self, metrics: Dict[str, float], mode: Literal["Train", "Validation"]) -> None:
        """
        Method that displays metrics into the terminal.

        :param Dict[str, float] **metrics**: Dictionary containing as keys metrics name and as items theirs values.
        :param Literal["Train", "Validation"] **mode**: String that can be equal to `Train` or `Validation`.
        """
        for metric_name, measure in metrics.items():
            print(f"{mode} {metric_name}: {round(measure, self._fp)}", end="|")
        print("\n")

    def _train(self, epoch: int) -> Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]]:
        """
        Method that trains the model.

        :param int **epoch**: Current training epoch.
        :return: Dictionaries containing the global average metrics and average metrics per class for the epoch.
        :rtype: Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]]
        """
        self._model.train()
        metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
        metrics["loss"] = []

        if self._detailed:
            metrics_per_class = {}
        else:
            metrics_per_class = None

        if self._verbose:
            loop = tqdm(self._train_loader, leave=True)
        else:
            loop = self._train_loader

        for x, y in loop:
            self._optimizer.zero_grad()

            x, y = x.to(self._device_model), y.to(self._device_model)
            y_pred = self._model(x)

            if self._mode == "object_detection":
                y_pred = y_pred.reshape(y.shape)

            loss = self._loss(y_pred, y)
            metrics["loss"].append(loss.item())

            if epoch != 0:
                loss.backward()
                self._optimizer.step()

            y, y_pred = self._prepare_gts_and_preds(y, y_pred)

            if y.shape[-1] != 1 and self._mode == "classification":
                y, y_pred = y.argmax(axis=-1), y_pred.argmax(axis=-1)

            if self._mode == "object_detection":
                y_pred, y = get_bboxes(x.shape[0], self._model.S, self._model.B, self._model.C, y_pred, y, iou_threshold=self._iou_threshold_overlay, confidence_threshold=self._confidence_threshold)

            for metric in self._metrics_to_use:
                measure = metric(y, y_pred)
                if isinstance(measure, tuple):
                    metrics[metric.name].append(measure[0])
                else:
                    metrics[metric.name].append(measure)

                if self._detailed and isinstance(measure, tuple):
                    if metric.name not in metrics_per_class.keys():
                        metrics_per_class[metric.name] = dict([(label, []) for label in range(self._num_classes)])

                    for label, value in measure[1].items():
                        metrics_per_class[metric.name][label].append(value)

            if self._verbose:
                loop.set_postfix([("Train_" + name, values[-1]) for name, values in metrics.items()])

        self._average(metrics, metrics_per_class)

        if self._detailed:
            return metrics, metrics_per_class
        else:
            return metrics
                
    def _validation(self) -> Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]]:
        """
        Method that evaluates the model.

        :return: Dictionaries containing the global average metrics and average metrics per class for the epoch.
        :rtype: Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]]
        """
        self._model.eval()
        metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
        metrics["loss"] = []

        if self._detailed:
            metrics_per_class = {}
        else:
            metrics_per_class = None

        if self._verbose:
            loop = tqdm(self._validation_loader, leave=True)
        else:
            loop = self._validation_loader

        for x, y in loop:
            with torch.no_grad():
                x, y = x.to(self._device_model), y.to(self._device_model)
                y_pred = self._model(x)

                if self._mode == "object_detection":
                    y_pred = y_pred.reshape(y.shape)

                loss = self._loss(y_pred, y)
                metrics["loss"].append(loss.item())

                y, y_pred = self._prepare_gts_and_preds(y, y_pred)

                if y.shape[-1] != 1 and self._mode == "classification":
                    y, y_pred = y.argmax(axis=-1), y_pred.argmax(axis=-1)

                if self._mode == "object_detection":
                    y_pred, y = get_bboxes(x.shape[0], self._model.S, self._model.B, self._model.C, y_pred, y, iou_threshold=self._iou_threshold_overlay, confidence_threshold=self._confidence_threshold)

                for metric in self._metrics_to_use:
                    measure = metric(y, y_pred)
                    if isinstance(measure, tuple):
                        metrics[metric.name].append(measure[0])
                    else:
                        metrics[metric.name].append(measure)

                    if self._detailed and isinstance(measure, tuple):
                        if metric.name not in metrics_per_class.keys():
                            metrics_per_class[metric.name] = dict([(label, []) for label in range(self._num_classes)])

                        for label, value in measure[1].items():
                            metrics_per_class[metric.name][label].append(value)

                if self._verbose:
                    loop.set_postfix([("Validation_" + name, values[-1]) for name, values in metrics.items()])

        self._average(metrics, metrics_per_class)

        if self._detailed:
            return metrics, metrics_per_class
        else:
            return metrics
    
    def _average(self, metrics: Dict[str, List[float]], metrics_per_class: Union[Dict[str, Dict[int, List[float]]], None] = None) -> None:
        """
        Method that averages the metrics.

        :param Dict[str, List[float]] **metrics**: Metrics used during the training.
        """
        for metric_name, values in metrics.items():
            if len(values) != 0:
                metrics[metric_name] = sum(values) / len(values)
            else:
                metrics[metric_name] = 0

        for metric_name in metrics_per_class:
            for label, values in metrics_per_class[metric_name].items():
                if len(metrics_per_class[metric_name][label]) != 0:
                    metrics_per_class[metric_name][label] = sum(metrics_per_class[metric_name][label]) / len(metrics_per_class[metric_name][label])
                else:
                    metrics_per_class[metric_name][label] = 0

    def _prepare_gts_and_preds(self, ground_truths: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method that prepares the ground truths and the predictions in order to have same size for the measurements.

        :param Tensor **ground_truths**: Tensor that contains ground truths of the given batch.
        :param Tensor **predictions**: Tensor that contains model predictions of the given batch.

        :return: Ground Truths and model predictions for the given batch.
        :rtype: Tuple[Tensor, Tensor]
        """
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
            new_predictions = torch.zeros(shape).to(ground_truths.device)
            if shape[-1] == 1:
                for i in range(ground_truths.shape[0]):
                    new_predictions[i] = int(preds[i].item())
            else:
                for i in range(ground_truths.shape[0]):
                    new_predictions[i][int(preds[i].item())] = 1

            return ground_truths, new_predictions
        
    def _save_model(self, epoch, metric_value: Union[float, None], metric_name: Union[str, None], stage: Union[str, None]):
        """
        Method that saves the model during the training.

        :param int **epoch**: Iteration.
        :param Union[float, None] **metric_value**: Value of the metric.
        :param Union[str, None] **metric_name**: The metric that is used to determine the save.
        :param Union[str, None] **stage**: Stage (train, validation) that produces the metric.
        """
        self._model.save(self._save_path, "model_checkpoint")

        if metric_value is not None and metric_name is not None and stage is not None:
            with open(self._save_log_path, "a") as file:
                file.write(f"\nmodel_checkpoint_{epoch}: {metric_value:.4f} of {metric_name} for {stage} stage.")
        print("checkpoint " + str(epoch) + " saved!")

    def _save_metrics(self, metrics: Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]], stage: str, name: Union[str, None] = None) -> None:
        """
        Method that saves metrics into a pickle file.

        :param Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]] **metrics**: metrics to save.
        :param str **stage**: Stage (train, validation) that produces the metrics.
        :param str **name**: Extra string for the name file.
        """
        if name is None:
            name = ""
        else:
            name = f"_{name}"
    
        with open(os.path.join(self._save_path, f"{stage}{name}.pickle"), 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)