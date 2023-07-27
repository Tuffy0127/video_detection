from ultralytics import YOLO
import os

d = os.getcwd()

#####train
# 导入模型
model = YOLO('yolov8n.pt')

# 默认数据集，自动下载，数据集很大，从github下载
# model.train(epochs=5)

# 从指定数据集训练
model.train(data="/root/autodl-tmp/boar-finder-7/data.yaml", epochs=100)

#####


#####predict
# 选已经训练过的模型
# model = YOLO(d + "/runs/detect/animals34_90/weights/best.pt")

# 视频格式
# model("test.mp4")

# 图片格式，save为True时，保存predict结果在runs中
# source = "/root/autodl-tmp/boar-finder-5/valid/images/481db3c9bd_jpg.rf.ae1f8d055f72e4ed2cc42a677e3c2786.jpg"
# results = model(source, save=True)

#####
