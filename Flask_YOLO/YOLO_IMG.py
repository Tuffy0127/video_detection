from ultralytics import YOLO
import cv2 as cv
import numpy as np
import math
import json


def img_detection(img):


    model = YOLO("C:/Users/leesh/Desktop/animal_recognizer/model/default.pt")
    result_list = []
    results = model(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = model.model.names[int(box.cls[0])]
            res = {'x1':x1,'y1':y1,'x2':x2,'y2':y2,'conf':conf,'cls':cls}
            result_list.append(res)

    return json.dumps(result_list,ensure_ascii=False,indent=2)
