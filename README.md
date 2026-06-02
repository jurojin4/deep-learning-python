# Deep Learning Python

This repository contains all my implementations in the field of **deep
learning** using Python as the programming language. However my goal, as future
AI engineer/AI researcher, is to use or develop mathematical tools in order
to provide solid explanations in this field, where many results are empirical.

## Installation with Conda
```
git clone https://github.com/jurojin4/deep-learning-python.git
cd deep-learning-python
```

```
conda create -n deep-learning-py python=3.13.8 - y
conda activate deep-learning-py
```

```
pip install -r requirements.txt
```


<!-- ## Download Datasets -->

<!-- ## Download Models weights -->

## Usage
### - Training:

```
python -m deep_learning_python.computer_vision.yolo.yolov1.main --dataset_name "pascalvoc2012" --dataset_path "dataset_path" --epochs 135 --image_size "(224,224)" --batch_size 32 --warmup_epoch 10 --save --save_metric "mAP@50"
python -m deep_learning_python.computer_vision.yolo.yolov3.main --dataset_name "pascalvoc2012" --dataset_path "dataset_path" --epochs 135 --image_size "(224,224)" --batch_size 32 --warmup_epoch 10 --save --save_metric "mAP@50"
```

### - Realtime computer vision (Torch):


```
python -m deep_learning_python.computer_vision.yolo.yolov1.generate_video --weights_path "weights_path" --video_path "video_path"
python -m deep_learning_python.computer_vision.yolo.yolov3.generate_video --weights_path "weights_path" --video_path "video_path"
```

### - Video generation:

```
python -m deep_learning_python.computer_vision.yolo.yolov1.generate_video --weights_path "weights_path" --video_path "video_path"
python -m deep_learning_python.computer_vision.yolo.yolov3.generate_video --weights_path "weights_path" --video_path "video_path"
```

### - Export PyTorch model to ONNX

```
python -m deep_learning_python.computer_vision.yolo.yolov6.export --weights_path "weights_path" --images_directory "images_directory_path"
```