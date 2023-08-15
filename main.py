import os
import supervision as sv
import cv2 as cv
from ultralytics import YOLO
import time


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
    model = YOLO(d + "/model/trail.pt")

    # 获取视频的宽和高
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # 视频写出
    # 文件 格式 帧率 分辨率 彩色
    writer = cv.VideoWriter(d + '/Video/output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (width, height), True)

    # 标记框参数
    box_annotator = sv.BoxAnnotator(
        thickness=2,  # 框线宽度
        text_thickness=2,  # 文字粗细
        text_scale=1  # 文字大小
    )

    counter = 0
    total_time = 0

    while True:
        start = time.time()
        # 切片
        ret, frame = cap.read()

        # 判断是否结束
        if not ret:
            break

        # 模型预测，agnostic参数 True表示多个类一起计算nms，False表示按照不同的类分别进行计算nms
        # CPU：device = 'cpu'，GPU：device = 0 / device = [0,1]
        result = model(frame, agnostic_nms=True, device=0)[0]

        # 预测结果存入supervision
        #  xyxy：boxes坐标array  mask  confidence：boxes对应物体置信度  class_id：boxes对应类编号  tracker_id
        detections = sv.Detections.from_yolov8(result)

        # model.model.names[id]，返回编号对应物品类的名称
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]

        # 用BoxAnnotator在帧中框出检测物品
        frame = box_annotator.annotate(
            scene=frame,  # 帧作为框选背景
            detections=detections,  # detections获取boxes
            labels=labels  # labels为boxes信息
        )

        # 写入帧 注掉可以提高速度
        # writer.write(frame)

        # 输出
        cv.imshow("yolov8", frame)

        # 每帧时长 输入esc退出
        if cv.waitKey(10) == 27:
            break

        end = time.time()
        print(end-start)
        total_time = total_time + end - start
        counter += 1

    # 释放
    cap.release()
    writer.release()
    cv.destroyAllWindows()

    print(counter/total_time)

    return


if __name__ == '__main__':
    animal_recognizer("video")
    # animal_recognizer("camera")
