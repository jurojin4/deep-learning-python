from trainer import train_model

import os

dirname = os.path.dirname(__file__)

if __name__ == "__main__":
    train_model(dataset_name="face_detection",
                dataset_path="",
                epochs=135,
                size=64,
                batch_size=80,
                learning_rate=1e-5,
                milestones=None,
                detailed=True,
                save=False,
                save_metric="loss",
                delete=False,
                weights_path=None,
                load_all=False,
                experiment_name="")
