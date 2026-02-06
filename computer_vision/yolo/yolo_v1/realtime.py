from PIL import Image
from yolov1 import YOLOV1
from prepare_dataset import Compose
from yolo_tools import get_bboxes_preds

import os
import time
import torch
import pickle
import cv2 as cv
import torchvision.transforms as transforms

def realtime(weights_path) -> None:
    """
    Method that uses the available camera on the computer to detect objects with YOLOV1 model.

    :param str **weights_path**: Weights path of the YOLOV1 model.
    """
    with open(os.path.join(os.path.dirname(weights_path), "checkpoint_log.txt"), "r") as file:
        info = file.readline().split("|")

    with open(os.path.join(os.path.dirname(weights_path), "categories.pickle"), "rb") as file:
        categories = pickle.load(file)

    categories = dict([(value, key) for key, value in categories.items()])
    mode = info[1].split(":")[1]
    num_classes = int(info[3].split(":")[1])
    size = int(info[4].split(":")[1].split(",")[0][1:])
    iou_threshold_overlay = float(info[11].split(":")[1])
    conf_threshold = float(info[12].split(":")[1])

    model = YOLOV1(in_channels=3, img_size=size, C=num_classes, batch_normalization=True, mode=mode)

    if weights_path is not None:
        model.load(weights_path=weights_path, all=True)
    print(f"Model parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    compose = Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean=0, std=255)])

    vid = cv.VideoCapture(0) 

    frame_width = 480
    frame_height = 640
    vid.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    prev_frame_time = 0
    new_frame_time = 0

    while(True):
        ret, frame = vid.read()
        new_frame_time = time.time()

        H,W,_ = frame.shape

        with torch.no_grad():
            x = compose(Image.fromarray(frame))
            x = x.unsqueeze(0)
            x = x.to(device)

            y_pred = model(x)
            y_pred = y_pred.reshape((1, model.S, model.S, model.C + (model.B * 5)))
            pred_bboxes = get_bboxes_preds(batch=1, S=model.S, B=model.B, C=model.C, predictions=y_pred, iou_threshold=iou_threshold_overlay, confidence_threshold=conf_threshold)
            for label, bboxes in enumerate(pred_bboxes):
                for box in bboxes:
                    objectiveness = round(float(box[2]), 3)
                    x1 = int(box[3] * W)
                    y1 = int(box[4] * H)
                    x2 = int((box[5] + box[3]) * W)
                    y2 = int((box[6] + box[4]) * H)
                    cv.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0), 3)
                    cv.putText(frame, categories[label] + f" {objectiveness}", (x1 - 5, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv.LINE_AA)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            cv.putText(frame, f"{int(fps)}", (7, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
            cv.imshow("YOLOV1 VOC2012 (20 classes)", frame)

            if cv.waitKey(1) & 0xFF == ord('q'): 
                break
    
    vid.release() 
    cv.destroyAllWindows()

if __name__ == "__main__":
    weights_path = "/home/otokonokage/Documents/github/deep-learning-py/computer_vision/yolo/yolo_v1/model_saves/pascalvoc2012/YOLOV1_object_detection_2025_350_22_13_26/model_checkpoint.pth"
    realtime(weights_path)