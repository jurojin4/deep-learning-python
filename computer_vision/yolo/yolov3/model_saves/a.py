import os
import pickle

dirname = os.path.dirname(__file__)

for dataset_directory in os.listdir(dirname):
    if not dataset_directory.endswith(".py"):
        for run in os.listdir(os.path.join(dirname, dataset_directory)):
            print(os.path.join(dirname, dataset_directory, run, "logs.pickle"))
            with open(os.path.join(dirname, dataset_directory, run, "logs.pickle"), "rb") as file:
                logs = pickle.load(file)

            if "data_augmentaion" in logs:
                data_augmentation = logs["data_augmentaion"]
                logs.pop('data_augmentaion', None)

                logs["data_augmentation"] = data_augmentation
            else:
                logs["data_augmentation"] = False
            with open(os.path.join(dirname, dataset_directory, run, "logs.pickle"), "wb") as file:
                pickle.dump(logs, file, protocol=pickle.HIGHEST_PROTOCOL)
