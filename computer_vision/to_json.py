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