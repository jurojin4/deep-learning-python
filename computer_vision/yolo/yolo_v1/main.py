from .trainer import YOLOV1_Trainer

import os
import argparse

dirname = os.path.dirname(__file__)

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOV1 Trainer', add_help=add_help)
    parser.add_argument("--dataset_name", default="pascalvoc2012", type=str, help='datasets available: ["cifar10", "face_detection", "pascalvoc2007", "pascalvoc2012", "tiny-imagenet200"]')
    parser.add_argument("--dataset_path", default=None, type=str, help="directory where the dataset is")
    parser.add_argument("--epochs", default=135, type=int, help="number of epochs during training")
    parser.add_argument("--size", default=224, type=int, help="Size of dataset images")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size during training")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="learning rate during training")
    parser.add_argument("--milestones", default=None, type=list, help="")
    parser.add_argument("--detailed", default=True, type=bool, help="extra details for metrics")
    parser.add_argument("--save", default=False, type=bool, help="save model during training")
    parser.add_argument("--save_metric", default="loss", type=str, help="metric used for model save")
    parser.add_argument("--delete", default=False, type=bool, help="delete dataset if it has been downloaded")
    parser.add_argument("--weights_path", default=None, type=str, help="")
    parser.add_argument("--load_all", default=False, type=bool, help="")
    parser.add_argument("--experiment_name", default=None, type=str, help="Name of the experiment")
    parser.add_argument("--verbose", default=True, type=bool, help="detailed training on terminal")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    trainer = YOLOV1_Trainer(dataset_name=args.dataset_name,
                dataset_path=args.dataset_path,
                epochs=args.epochs,
                size=args.size,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                milestones=args.milestones,
                detailed=args.detailed,
                save=args.save,
                save_metric=args.save_metric,
                delete=args.delete,
                weights_path=args.weights_path,
                load_all=args.load_all,
                experiment_name=args.experiment_name,
                verbose=args.verbose)
    trainer()