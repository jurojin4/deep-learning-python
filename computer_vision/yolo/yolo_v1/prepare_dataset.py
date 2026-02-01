from tqdm import tqdm
from PIL import Image
from typing import Callable, Dict, List, Literal, Tuple, Union
from torch.utils.data import Dataset

import os
import json
import torch
import pickle
import subprocess
import torchvision
import numpy as np
import xml.etree.ElementTree as ET

def prepare_tiny_imagenet(path: str, delete: bool = False) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[int]], list[float], int, Dict[str, int], Literal['tiny-imagenet200'], Literal['classification']]:
    """
    Method that prepares the "classification" tiny-imagenet200 dataset.

    :param str **path**: Path where the dataset will be stored after the download.
    :param bool **delete**: Boolean that allows to delete the dataset at the end of the preparation. Set to `False`.
    :return: Train set, validation set, weights, number of classes, classes with their id, dataset name and dataset type.
    :rtype: Tuple[Dict[str, List[np.ndarray]], Dict[str, List[int]], list[float], int, Dict[str, int], Literal['tiny-imagenet200'], Literal['classification']]
    """
    assert isinstance(delete, bool), f"delete has to be an {bool} instance, not {type(delete)}."
    if not os.path.exists(os.path.join(path, "tiny-imagenet-200")):
        zip_path = os.path.join(path, "tiny-imagenet-200.zip")

        os.makedirs(path, exist_ok=True)

        subprocess.run(["wget", "-O", zip_path, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])
        subprocess.run(["unzip", zip_path, "-d", path])

        os.remove(zip_path)

    dataset_path = os.path.join(path, "tiny-imagenet-200")

    _categories = {}
    with open(os.path.join(dataset_path, "wnids.txt"), "r") as file:
        for enu, line in enumerate(file.readlines()):
            _categories[line.split("\n")[0]] = enu

    categories = {}
    with open(os.path.join(dataset_path, "words.txt"), "r") as file:
        for line in file.readlines():
            split = line.split("	")
            label_id, label_name = split[0], split[1]
            if label_id in _categories:
                categories[label_name.replace("\n", "")] = _categories[label_id]

    # Train
    _n_total_imgs = 0

    weights = []
    train = {"images": [], "labels": []}
    for enu, category in enumerate(tqdm(os.listdir(os.path.join(dataset_path, "train")), leave=True)):
        with open(os.path.join(dataset_path, "train", category, f"{category}_boxes.txt"), "r") as file:
            _bboxes_img = file.readlines()

        weights.append(0)
        for bbox_img in _bboxes_img:
            image_name = bbox_img.replace("\t", " ").split(" ")[0]
            image = Image.open(os.path.join(dataset_path, "train", category, "images", image_name), mode="r").convert("RGB")
            train["images"].append(np.array(image))
            train["labels"].append(_categories[category])

            weights[enu] += 1
            _n_total_imgs += 1
    
    for i in range(len(weights)):
        weights[i] /= _n_total_imgs

    # Validation
    validation = {"images": [], "labels": []}
    with open(os.path.join(dataset_path, "val", "val_annotations.txt"), "r") as file:
        _bboxes_img = file.readlines()

    for bbox_img in tqdm(_bboxes_img, leave=True):
        _info = bbox_img.replace("\t", " ").split(" ")
        image_name, category = _info[0], _info[1]
        image = Image.open(os.path.join(dataset_path, "val", "images", image_name), mode="r").convert("RGB")
        validation["images"].append(np.array(image))
        validation["labels"].append(_categories[category])
            
    if delete:
        os.remove(os.path.join(path, "tiny-imagenet-200"))

    return train, validation, weights, len(categories), categories, "tiny-imagenet200", "classification"

def prepare_cifar10(path: str, delete: bool = False) -> Tuple[dict[str, List[np.ndarray]], dict[str, List[int]], list[float], int, Dict[str, int], Literal['cifar-10'], Literal['classification']]:
    """
    Method that prepares the "classification" tiny-imagenet200 dataset.

    :param str **path**: Path where the dataset will be stored after the download.
    :param bool **delete**: Boolean that allows to delete the dataset at the end of the preparation. Set to `False`.
    :return: Train set, validation set, weights, number of classes, classes with their id, dataset name and dataset type.
    :rtype: Tuple[dict[str, List[np.ndarray]], dict[str, List[int]], list[float], int, Dict[str, int], Literal['cifar-10'], Literal['classification']]
    """
    assert isinstance(delete, bool), f"delete has to be an {bool} instance, not {type(delete)}."

    torchvision.datasets.CIFAR10(root=path, download=True)
    try:
        os.remove(os.path.join(path, "cifar-10-python.tar.gz"))
    except:
        pass

    train = {"images": [], "labels": []}
    validation = {"images": [], "labels": []}
    
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding="bytes")
        
        return dict

    dataset_path = os.path.join(path, "cifar-10-batches-py")

    categories = dict([(category.decode("utf-8"), enu) for enu, category in enumerate(unpickle(os.path.join(dataset_path, "batches.meta"))[b"label_names"])])
    num_classes = len(unpickle(os.path.join(dataset_path, "batches.meta"))[b"label_names"])

    _n_total_imgs = 0

    for file in tqdm(os.listdir(dataset_path), leave=True):
        add_train = True

        if file.find(".") != -1:
            continue

        if file.find("test") != -1:
            add_train = False
            
        d = unpickle(os.path.join(dataset_path, file))

        weights = [0] * num_classes

        _n_total_imgs += 1
            
        for data, label in zip(d[bytes("data", encoding="utf-8")], d[bytes("labels", encoding="utf-8")]):
            R = data[:1024].reshape(32, 32, 1)
            G = data[1024:1024*2].reshape(32, 32, 1)
            B = data[1024*2:1024*3].reshape(32, 32, 1)

            image = np.concat((R, G, B), axis=-1)
            if add_train:
                train["images"].append(image)
                train["labels"].append(label)
                weights[label] += 1
            else:
                validation["images"].append(image)
                validation["labels"].append(label)

    for i in range(len(weights)):
        weights[i] /= _n_total_imgs

    if delete:
        os.remove(os.path.join(path, "cifar-10-batches-py"))
    return train, validation, weights, num_classes, categories, "cifar-10", "classification"

def prepare_vocdetection(path: str, year: Literal["2007", "2012"] = "2007", box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh") -> Union[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['pascalvoc2007'], Literal['object_detection']], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['pascalvoc2012'], Literal['object_detection']]]:
    """
    Method that prepares the "classification" tiny-imagenet200 dataset.

    :param str **path**: Path where the dataset is stored.
    :param Literal["2007", "2012"] **year**: Version of the dataset.
    :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `"xywh"`.
    :return: Train set, validation set, weights, number of classes, classes with their id, dataset name and dataset type.
    :rtype: Union[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['pascalvoc2007'], Literal['object_detection']], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['pascalvoc2012'], Literal['object_detection']]]
    """
    assert year in ["2007", "2012"], f"year has to be in [\"2007\", \"2012\"]."
    assert box_format in ["xyxy", "xywh", "xcycwh"], f"year has to be in [\"xyxy\", \"xywh\", \"xcycwh\"]."

    if year == "2007":
        with open(os.path.join(path, "pascal_voc2007/pascal_voc/pascal_train2007.json"), "r") as file:
            train_data = json.load(file)
        with open(os.path.join(path, "pascal_voc2007/pascal_voc/pascal_val2007.json"), "r") as file:
            val_data = json.load(file)

        categories = dict([(c["name"], c["id"]) for c in train_data["categories"]])
        weights = [0] * len(categories)

        def _treatment_2007(data, update_weights: bool = True):
            images = {}
            bboxes = {}
            labels = {}
            for annotation_data in data["annotations"]:
                x, y, w, h = annotation_data["bbox"]

                if box_format == "xyxy":
                    bbox = [x, y, x + w, y + h]
                elif box_format == "xywh":
                    bbox = [x, y, w, h]
                elif box_format == "xcycwh":
                    bbox = [x + (w / 2), y + (h / 2), w, h]

                if annotation_data["image_id"] not in bboxes.keys():
                    bboxes[annotation_data["image_id"]] = [bbox]
                    labels[annotation_data["image_id"]] = [annotation_data["category_id"]-1]
                else:
                    bboxes[annotation_data["image_id"]].append(bbox)
                    labels[annotation_data["image_id"]].append(annotation_data["category_id"]-1)

                if update_weights:
                    weights[annotation_data["category_id"]-1] += 1

            images_path = os.path.join(path, "pascal_voc2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/")
            for img_id in bboxes:
                n = len(str(img_id))
                file = f"{"0" * (6 - n)}{img_id}.jpg"
                image = np.array(Image.open(os.path.join(images_path, file)))
                images[img_id] = image
                H, W, _ = image.shape

                boxes = bboxes[img_id]
                new_boxes = []
                for box in boxes:
                    new_boxes.append([box[0] / W, box[1] / H, box[2] / W, box[3] / H])

                bboxes[img_id] = new_boxes
                
            return {"images": images, "bboxes": bboxes, "labels": labels}
        return _treatment_2007(train_data), _treatment_2007(val_data, False), weights, len(categories), categories, f"pascalvoc{year}", "object_detection"
    else:
        with open(os.path.join(path, "VOC2012/ImageSets/Main/train.txt"), "r") as file:
            train_data = file.readlines()
        with open(os.path.join(path, "VOC2012/ImageSets/Main/val.txt"), "r") as file:
            val_data = file.readlines()

        train_data = [file.replace("\n", "") for file in train_data]
        val_data = [file.replace("\n", "") for file in val_data]

        categories = {}
        weights = [0] * 20
        def _treatment_2012(data, update_weights: bool = True):
            images = {}
            bboxes = {}
            labels = {}
            
            for file in tqdm(os.listdir(os.path.join(path, "VOC2012/Annotations/")), leave=True):
                if file[:-4] in data:
                    tree = ET.parse(os.path.join(path, "VOC2012/Annotations", file))
                    root = tree.getroot()

                    width = int(root.find("size/width").text)
                    height = int(root.find("size/height").text)

                    for obj in root.findall("object"):
                        label = obj.find("name").text
                        if label not in categories.keys():
                            categories[label] = len(categories)

                        bbox = obj.find("bndbox")
                        xmin = float(bbox.find("xmin").text)
                        ymin = float(bbox.find("ymin").text)
                        xmax = float(bbox.find("xmax").text)
                        ymax = float(bbox.find("ymax").text)

                        if box_format == "xyxy":
                            bbox = [xmin / width, ymin / height, xmax / width, ymax / height]
                        elif box_format == "xywh":
                            bbox = [xmin / width, ymin / height, (xmax - xmin) / width, (ymax - ymin) / height]
                        elif box_format == "xcycwh":
                            w = xmax - xmin
                            h = ymax - ymin
                            bbox = [(xmin + (w / 2)) / width, (ymin + (h / 2)) / height, w / width, h / height]

                        id = file[:-4].split("_")[1]
                        if id not in bboxes.keys():
                            bboxes[id] = [bbox]
                            labels[id] = [categories[label]]
                        else:
                            bboxes[id].append(bbox)
                            labels[id].append(categories[label])
                        
                        if update_weights:
                            weights[categories[label]] += 1

                    image = np.array(Image.open(os.path.join(path, "VOC2012/JPEGImages", f"{file[:-4]}.jpg")))
                    images[id] = image
            return {"images": images, "bboxes": bboxes, "labels": labels}

        return _treatment_2012(train_data), _treatment_2012(val_data, False), weights, len(categories), categories, f"pascalvoc{year}", "object_detection"

class Compose(object):
    """
    Compose class.
    """
    def __init__(self, transforms: List[Callable]):
        """
        Initializes the Compose class.
        
        :param List[Callable] **transforms**: Transformations to apply to objects (images)."""
        self.transforms = transforms

    def __call__(self, image: Union[np.ndarray, Image.Image]):
        """
        Built-in Python method that allows to call an instance of the class. Applies the transformations.
        
        :param Union[np.ndarray, Image.Image] **image**:
        :return: Image transformed.
        :rtype: ArrayLike
        """
        for t in self.transforms:
            image = t(image)
        return image

class YOLOV1Dataset(Dataset):
    """
    YOLOV1Dataset class.
    """
    def __init__(self, dataset: Union[Dict[str, List[np.ndarray]], Dict[str, List[int]], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]]]], C: int, S: int = 7, B: int = 2, mode: Literal["classification", "object_detection"] = "object_detection", compose: Union[Compose, None] = None):
        """
        Initializes the YOLOV1Dataset class.

        :param Union[Dict[str, List[np.ndarray]], Dict[str, List[int]], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]]]] **dataset**: Dataset used.
        :param int **C**: Number of classes to predict.
        :param int **S**: Cells number along height and width.
        :param int **B**: Bounding boxes number in each cell.
        :param Literal["classification", "object_detection"] **mode**: String that defines the model mode. Set to `object_detection`.
        :param Union[Compose, None] **compose**: Tool transforming images. Set to `None`.
        """
        assert isinstance(mode, str) and mode in ["classification", "object_detection"], f"mode has to be a {str} instance and in the given list:\n[\"classification\", \"object_detection\"]"
        assert isinstance(compose, Compose) or compose == None, f"compose has to be a {Compose} instance (or None), not {type(compose)}."
        super().__init__()
        self.C = C
        self._mode = mode
        self._compose = compose

        if mode == "classification":
            self._images = dataset["images"]
            self._labels = dataset["labels"]
            self.S = None
            self.B = None
        else:
            self._images = dataset["images"]
            self._bboxes = dataset["bboxes"]
            self._labels = dataset["labels"]
            self.S = S
            self.B = B

            self._type = isinstance(self._images, dict)
            if self._type:
                self._keys = list(self._images.keys())

    def __len__(self) -> int:
        """
        Built-in Python method that returns the length of the object.
        :return: Number of images in the dataset.
        :rtype: int
        """
        return len(self._images)
    
    def _loc(self, line: torch.Tensor, rate: int = 5) -> int:
        """
        Method that returns the location of the bounding box in the line of size C + B * 2 from case C.
        The method attributes one of B spot in the line, if none is empty then the value returned is superior than B * 5 + C.
        
        :param Tensor **line*: 
        :param int **rate**: Number of elements in a box.
        :return: Beginning of the box.
        :rtype: int
        """
        for i in range(self.C, line.shape[0], rate):
            if line[i] == 0:
                return int(i/rate)
        return int((i/rate)) + 1
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Built-in Python method that allows to access an element from the object.

        :param int **idx**:
        :return: In mode "classification", a tensor and a label.In mode "object_detection, a tuple of tensor of size 2.
        :rtype: Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
        """
        if self._mode == "classification":
            image = self._images[idx]
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if self._compose is not None:
                image = self._compose(image)
            
            return image, self._labels[idx]
        else:
            if self._type:
                idx = self._keys[idx]
                
            image = self._images[idx]
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if self._compose is not None:
                image = self._compose(image)
            
            grid_label = torch.zeros(size=(self.S, self.S, self.C + (5 * self.B)))

            bboxes = self._bboxes[idx]
            labels = self._labels[idx]
            for bbox, label in zip(bboxes, labels):
                x, y, w, h = bbox
                i, j = int(y * self.S), int(x * self.S)

                x_cell, y_cell = self.S * x - j, self.S * y - i
                height_cell, width_cell = (h * self.S, w * self.S)

                pos = torch.nonzero(grid_label[i, j, :self.C])
                if pos.nelement() != 0:
                    if pos[0].item() == label:
                        n = self._loc(grid_label[i,j])
                        if n * 5 < self.B * 5 + self.C:
                            grid_label[i, j, n * 5] = 1

                            box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                            grid_label[i, j, n * 5 + 1:5 * (n+1)] = box_coordinates
                else:
                    grid_label[i, j, self.C] = 1

                    cell_bbox = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    grid_label[i, j, self.C+1:self.C+5] = cell_bbox

                    grid_label[i, j, label] = 1
            
            return image, grid_label