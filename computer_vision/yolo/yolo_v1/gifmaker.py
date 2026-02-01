from tqdm import tqdm

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as io

dirname = os.path.dirname(__file__)

def make_gif(path: str) -> None:
    """
    Method that creates a gif from metrics during training.
    
    :param str **path**: Path of the metric.
    """
    experiments = {}

    def load_pickle(path):
        with open(path, "rb") as file:
            return pickle.load(file)
        
    epochs = 0

    for dir in os.listdir(path):
        if not dir.endswith(".gif"):
            dir_path = os.path.join(path, dir)
        else:
            continue

        with open(os.path.join(dir_path, "checkpoint_log.txt"), "r") as file:
            info = file.readline().split("|")
        
        if epochs is None:
            epochs = int(info[6].split(":")[1])
        else:
            if epochs < int(int(info[6].split(":")[1])):
                epochs = int(int(info[6].split(":")[1]))

        experiment_name = info[0].split(":")[1]
        experiments[experiment_name] = {}

        categories = load_pickle(os.path.join(dir_path, "categories.pickle"))
        experiments[experiment_name]["categories"] = categories

        train_d = load_pickle(os.path.join(dir_path, "train.pickle"))
        experiments[experiment_name]["train_metrics"] = train_d

        if os.path.exists(os.path.join(dir_path, "train_pc.pickle")):
            train_dpc = load_pickle(os.path.join(dir_path, "train_pc.pickle"))
            experiments[experiment_name]["train_metrics_pc"] = train_dpc

        if os.path.exists(os.path.join(dir_path, "validation.pickle")):
            validation_d = load_pickle(os.path.join(dir_path, "validation.pickle"))
            experiments[experiment_name]["validation_metrics"] = validation_d
            
            if os.path.exists(os.path.join(dir_path, "validation_pc.pickle")):
                validation_dpc = load_pickle(os.path.join(dir_path, "validation_pc.pickle"))
                experiments[experiment_name]["validation_metrics_pc"] = validation_dpc

    train_detailed = False
    validation_detailed = False
    has_validation = False
    for name in experiments:
        if "validation_metrics" in experiments[name]:
            has_validation = True

        if not train_detailed:
            train_detailed = "train_metrics_pc" in experiments[name].keys()

        if not validation_detailed:
            validation_detailed = "validation_metrics_pc" in experiments[name].keys()

    fig, axes = plt.subplots(2, len(experiments[name]["train_metrics"]), figsize=(19.2, 10.8))
    boolean = True if len(experiments[name]["train_metrics"]) > 1 else False

    fig.tight_layout(h_pad=5, w_pad=2)
    images_metrics = []

    #print(dict(sorted([(experiment_name, stage) for experiment_name, stage in experiments.items()], key=lambda x: x[0].lower())).keys())

    if train_detailed:
        fig_pc, axes_pc = plt.subplots(2, len(experiments[name]["train_metrics_pc"]), figsize=(19.2, 10.8))

        boolean_pc = True if len(experiments[name]["train_metrics_pc"]) > 1 else False

        fig_pc.tight_layout(h_pad=5, w_pad=2)
        fig_pc.subplots_adjust(top=0.925)

        images_pc = [[] for _ in range(len(experiments))]

    for epoch in tqdm(range(0, epochs + 1), leave=True):
        for _axes in axes:
            try:
                for ax in _axes:
                    ax.clear()
                    ax.set_xlim(0, epochs)
            except:
                boolean = False
                _axes.clear()
                _axes.set_xlim(0, epochs)

        for experiment_name in experiments:
            for enum, train_metric in enumerate(experiments[experiment_name]["train_metrics"]):
                if has_validation and boolean:
                    axes[0][enum].plot(range(epochs + 1), experiments[experiment_name]["train_metrics"][train_metric][:epoch + 1] + [np.nan] * (epochs + 1  - len(experiments[experiment_name]["train_metrics"][train_metric][:epoch + 1])), label=experiment_name)
                    axes[0][enum].set_title(f"Train {train_metric}")
                    axes[0][enum].legend()
                else:
                    axes[enum].plot(range(epochs + 1), experiments[experiment_name]["train_metrics"][train_metric][:epoch + 1] + [np.nan] * (epochs + 1 - len(experiments[experiment_name]["train_metrics"][train_metric][:epoch + 1])), label=experiment_name)
                    axes[enum].set_title(f"Train {train_metric}")
                    axes[enum].legend()

            if "validation_metrics" in experiments[experiment_name]:
                for enum, validation_metric in enumerate(experiments[experiment_name]["validation_metrics"]):
                    if boolean:
                        axes[1][enum].plot(range(epochs + 1), experiments[experiment_name]["validation_metrics"][validation_metric][:epoch + 1] + [np.nan] * (epochs + 1 - len(experiments[experiment_name]["validation_metrics"][validation_metric][:epoch + 1])), label=experiment_name)
                        axes[1][enum].set_title(f"Validation {validation_metric}")
                        axes[1][enum].legend()
                    else:
                        axes[enum].plot(range(epochs + 1), experiments[experiment_name]["validation_metrics"][validation_metric][:epoch + 1] + [np.nan] * (epochs + 1 - len(experiments[experiment_name]["validation_metrics"][validation_metric][:epoch + 1])), label=experiment_name)
                        axes[enum].set_title(f"Validation {validation_metric}")
                        axes[enum].legend()

        fig.canvas.draw()
        images_metrics.append(np.array(fig.canvas.renderer.buffer_rgba()))

        if train_detailed:

            random_key = list(experiments[experiment_name]["train_metrics_pc"])[0]
            random_key = list(experiments[experiment_name]["train_metrics_pc"][random_key])[0]

            fig_pc.suptitle(f"Epoch: {epoch} / {epochs}", fontsize=18)

            for enu, experiment_name in enumerate(experiments):
                for _axes in axes_pc:
                    try:
                        for ax in _axes:
                            ax.clear()
                            ax.set_ylim(0, 1)
                    except:
                        boolean_pc = False
                        _axes.clear()
                        _axes.set_ylim(0, 1)

                for enum, train_metric in enumerate(experiments[experiment_name]["train_metrics_pc"]):
                    if boolean_pc:
                        if len(experiments[experiment_name]["train_metrics_pc"][train_metric][random_key]) > epoch:
                            axes_pc[0][enum].bar([key for key, value in categories.items()], [values[epoch] for _, values in experiments[experiment_name]["train_metrics_pc"][train_metric].items()])
                            axes_pc[0][enum].set_title(f"Train {train_metric} per class")
                        else:
                            axes_pc[0][enum].bar([key for key, value in categories.items()], [values[len(values) - 1] for _, values in experiments[experiment_name]["train_metrics_pc"][train_metric].items()])
                            axes_pc[0][enum].set_title(f"Train {train_metric} per class")
                    else:
                        if len(experiments[experiment_name]["train_metrics_pc"][train_metric][random_key]) > epoch:
                            axes_pc[0].bar([key for key, value in categories.items()], [values[epoch] for _, values in experiments[experiment_name]["train_metrics_pc"][train_metric].items()])
                            axes_pc[0].set_title(f"Train {train_metric} per class")
                        else:
                            axes_pc[0].bar([key for key, value in categories.items()], [values[len(values) - 1] for _, values in experiments[experiment_name]["train_metrics_pc"][train_metric].items()])
                            axes_pc[0].set_title(f"Train {train_metric} per class")

                if "validation_metrics_pc" in experiments[experiment_name]:
                    for enum, validation_metric in enumerate(experiments[experiment_name]["validation_metrics_pc"]):
                        if boolean_pc:
                            if len(experiments[experiment_name]["validation_metrics_pc"][validation_metric][random_key]) > epoch:
                                axes_pc[1][enum].bar([key for key, value in categories.items()], [values[epoch] for _, values in experiments[experiment_name]["validation_metrics_pc"][validation_metric].items()])
                                axes_pc[1][enum].set_title(f"Validation {validation_metric} per class")
                            else:
                                axes_pc[1][enum].bar([key for key, value in categories.items()], [values[len(values) - 1] for _, values in experiments[experiment_name]["validation_metrics_pc"][validation_metric].items()])
                                axes_pc[1][enum].set_title(f"Validation {validation_metric} per class")
                        else:
                            if len(experiments[experiment_name]["validation_metrics_pc"][validation_metric][random_key]) > epoch:
                                axes_pc[1].bar([key for key, value in categories.items()], [values[epoch] for _, values in experiments[experiment_name]["validation_metrics_pc"][validation_metric].items()])
                                axes_pc[1].set_title(f"Validation {validation_metric} per class")
                            else:
                                axes_pc[1].bar([key for key, value in categories.items()], [values[len(values) - 1] for _, values in experiments[experiment_name]["validation_metrics_pc"][validation_metric].items()])
                                axes_pc[1].set_title(f"Validation {validation_metric} per class")

                fig_pc.canvas.draw()
                images_pc[enu].append(np.array(fig_pc.canvas.renderer.buffer_rgba()))

    io.mimsave(os.path.join(path, "metrics.gif"), images_metrics, duration=1)

    for enu, experiment_name in enumerate(tqdm(experiments, leave=True)):
        io.mimsave(os.path.join(path, f"{experiment_name}_metrics_per_class.gif"), images_pc[enu], duration=1)

if __name__ == "__main__":
    path = os.path.join(dirname, "model_saves", "pascalvoc2012")
    make_gif(path)