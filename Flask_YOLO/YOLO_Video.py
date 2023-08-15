import sys
sys.path.append("..")
from ultralytics import YOLO
import cv2 as cv
import math
import os
import supervision as sv


def video_detection(path_x):
    video_capture = path_x
    # d = os.getcwd()


    cap = cv.VideoCapture(video_capture)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("C:/Users/leesh/Desktop/animal_recognizer/model/default.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,  # 框线宽度
        text_thickness=2,  # 文字粗细
        text_scale=1  # 文字大小
    )

    while True:
        ret, img = cap.read()

        if not ret:
            break

        result = model(img, agnostic_nms=True, device=0)[0]

        detections = sv.Detections.from_yolov8(result)

        # model.model.names[id]，返回编号对应物品类的名称
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]

        # 用BoxAnnotator在帧中框出检测物品
        frame = box_annotator.annotate(
            scene=img,  # 帧作为框选背景
            detections=detections,  # detections获取boxes
            labels=labels  # labels为boxes信息
        )

    yield img

cv.destroyAllWindows()


