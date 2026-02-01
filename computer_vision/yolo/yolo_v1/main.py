from trainer import train_model

import os

dirname = os.path.dirname(__file__)

if __name__ == "__main__":
    train_model(dataset_name="pascalvoc2012",
                dataset_path="/home/otokonokage/Documents/github/dataset/computer_vision/",
                epochs=135,
                size=224,
                batch_size=32,
                learning_rate=1e-5,
                milestones=None,
                detailed=True,
                save=True,
                save_metric="loss",
                delete=False,
                weights_path=None, #"/home/otokonokage/Documents/github/deep-learning-py/computer_vision/yolo/yolo_v1/model_saves/tiny-imagenet200/YOLOV1_classification_2025_360_16_20_29/model_checkpoint.pth",
                load_all=False,
                experiment_name="No batch normalization in CNN block")
