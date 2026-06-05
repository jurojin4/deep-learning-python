import os
import json
import pickle

def to_json(dirpath: str):
    for filename in os.listdir(dirpath):
        if filename.endswith(".pickle"):
            with open(os.path.join(dirpath, filename), "rb") as file:
                dictionary = pickle.load(file)

            with open(os.path.join(dirpath, filename.split(".")[0] + ".json"), "w") as file:
                json.dump(dictionary, file)

if __name__ == "__main__":
    to_json("/home/otokonokage/Documents/github/deep_learning_python/computer_vision/yolo/yolov6/model_saves/modified_coco2017/asu-4862/")