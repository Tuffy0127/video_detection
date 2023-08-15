import os
import supervision as sv
import cv2 as cv
from ultralytics import YOLO
import time
from pathlib import Path
import numpy as np
from typing import Tuple,Dict
from ultralytics.yolo.utils import ops
import torch
from openvino.runtime import Core, Model
import random
from ultralytics.yolo.utils.plotting import colors
from PIL import Image


def plot_one_box(box: np.ndarray, img: np.ndarray, color: Tuple[int, int, int] = None, mask: np.ndarray = None,
                 label: str = None, line_thickness: int = 5):

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3])) # box: sx sy ex ey
    cv.rectangle(img, c1, c2, color, thickness=tl, lineType=cv.LINE_AA) # cv.LINE_AA 抗锯齿线
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv.rectangle(img, c1, c2, color, -1, cv.LINE_AA)  # filled 文字框
        cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA) #文字
    if mask is not None:
        image_with_mask = img.copy()
        mask
        cv.fillPoly(image_with_mask, pts=[mask.astype(int)], color=color)
        img = cv.addWeighted(img, 0.5, image_with_mask, 0.5, 1)
    return img

def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114),
              auto: bool = False, scale_fill: bool = False, scaleup: bool = False, stride: int = 32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess_image(img0: np.ndarray):
    # resize
    img = letterbox(img0)[0]

    # Convert HWC to CHW
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img) # cv2使用方法imread读出来图片的数据是[h w c] ,但是在pytorch中，我们往往使用的图片数据结构是[c h w]，
    return img


def image_to_tensor(image: np.ndarray):
    input_tensor = image.astype(np.float32)  # uint8 to fp32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    # add batch dimension
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


def postprocess(
        pred_boxes: np.ndarray,
        input_hw: Tuple[int, int],
        orig_img: np.ndarray,
        min_conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.7,
        agnosting_nms: bool = False,
        max_detections: int = 300,
        pred_masks: np.ndarray = None,
        retina_mask: bool = False
):
    nms_kwargs = {"agnostic": agnosting_nms, "max_det": max_detections}
    # if pred_masks is not None:
    #     nms_kwargs["nm"] = 32
    preds = ops.non_max_suppression(
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        nc=80,
        **nms_kwargs
    )
    results = []
    proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "segment": []})
            continue
        if proto is None:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
            continue
        if retina_mask:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        results.append({"det": pred[:, :6].numpy(), "segment": segments})
    return results


def detect(image: np.ndarray, model: Model):
    num_outputs = len(model.outputs)
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    result = model(input_tensor)
    boxes = result[model.output(0)]
    masks = None
    if num_outputs > 1:
        masks = result[model.output(1)]
    input_hw = input_tensor.shape[2:]
    detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
    return detections


def draw_results(results: Dict, source_image: np.ndarray, label_map: Dict):
    boxes = results["det"]
    masks = results.get("segment")
    h, w = source_image.shape[:2]
    for idx, (*xyxy, conf, lbl) in enumerate(boxes):
        label = f'{label_map[int(lbl)]} {conf:.2f}'
        mask = masks[idx] if masks is not None else None
        source_image = plot_one_box(xyxy, source_image, mask=mask, label=label, color=colors(int(lbl)),
                                    line_thickness=1)
    return source_image


def animal_recognizer(t):
    d = os.getcwd()
    if t == "video":
        # 视频输入
        cap = cv.VideoCapture(d + "/Video/boar.mp4")
        cap.set(cv.CAP_PROP_FPS, 30)
    elif t == "camera":
        # 摄像头输入
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FPS, 30)
    else:
        print("invalid input")
        return

    # 载入训练完成的模型
    model = YOLO(d + "/model/default.pt")
    label_map = model.model.names

    DET_MODEL_NAME = "default"
    models_dir = Path('.')
    det_model_path = models_dir / f"model/{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
    if not det_model_path.exists():
        model.export(format="openvino", dynamic=True, half=False)

    core = Core()
    det_ov_model = core.read_model(det_model_path)
    device = "GPU"  # "GPU"
    if device != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    det_compiled_model = core.compile_model(det_ov_model, device)

    # 获取视频的宽和高
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # 视频写出
    # 文件 格式 帧率 分辨率 彩色
    # writer = cv.VideoWriter(d + '/Video/output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (width, height), True)

    counter = 0
    total_time = 0

    while True:
        start = time.time()
        # 切片
        ret, frame = cap.read()

        # 判断是否结束
        if not ret:
            break

        input_image = np.array(frame)
        start = time.time()
        detections = detect(input_image, det_compiled_model)[0]
        image_with_boxes = draw_results(detections, input_image, label_map)




        # 输出
        cv.imshow("yolov8", image_with_boxes)

        # 每帧时长 输入esc退出
        if cv.waitKey(10) == 27:
            break

        end = time.time()
        print(end - start)
        total_time = total_time + end - start
        counter += 1

    # 释放
    cap.release()
    # writer.release()
    cv.destroyAllWindows()

    print(counter / total_time)

    return


if __name__ == '__main__':
    animal_recognizer("video")
    # animal_recognizer("camera")
