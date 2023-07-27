# YOLOv8

[GitHub - ultralytics/ultralytics: NEW - YOLOv8 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/ultralytics)

[TOC]



## 环境配置

### Anaconda

 创建anaconda虚拟环境

```
conda create -n pyhton3.10 python=3.10
conda activate 
```

### pytorch和torchvision

[PyTorch](https://pytorch.org/)

在pytorch官网查找cuda版本对应的pip或conda指令，如果不需要GPU训练和预测，直接pip或conda也可以

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### YOLOv8(ultralytics)

[GitHub - ultralytics/ultralytics: NEW - YOLOv8 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/ultralytics)

这里建议使用

```
git clone https://github.com/ultralytics/ultralytics
```

直接zip下载安装会导致运行时出现报错

```
ImportError: Bad git executable
```

在ultralytics（第一个）目录下

通过pip安装requirements

```
pip install -r requirements.txt
```

（注意：这里直接安装的Pillow版本过高可能导致错误，我手动降到Pillow=9.2.0。如果用AutoDL，环境都是配好的，直接用就可以。）

## 数据准备

### YOLOv8自带数据

训练时不注明具体的data.yaml文件（或是使用'coco128.yaml'），则会自动在github下载数据集。默认数据集很大，github没有科学上网下载很慢。（😭我™30个G的梯子流量）

coco128这个数据集不是很大。

### roboflow数据集

[Roboflow](https://app.roboflow.com/)

roboflow是一个很好用的标记数据的网站。创建项目后自己上传数据图片，添加标记之后就可以开始标注。

![image-20230713115104273](C:\Users\leesh\AppData\Roaming\Typora\typora-user-images\image-20230713115104273.png)

选择文件或者文件夹上传图片 => 选择Save and Continue继续 => 选择团队内成员分配工作

开始标注

![image-20230713141000137](C:\Users\leesh\AppData\Roaming\Typora\typora-user-images\image-20230713141000137.png)

标注完成后添加到数据库中，roboflow会自动对数据分类。

![image-20230713141131295](C:\Users\leesh\AppData\Roaming\Typora\typora-user-images\image-20230713141131295.png)

最后在generate中导出训练数据。这里可以选择对数据集图片进行resize等预处理，YOLOv8文档里推荐使用的是Auto-Orient，和640*640的resize。

![image-20230713141255580](C:\Users\leesh\AppData\Roaming\Typora\typora-user-images\image-20230713141255580.png)



最后选择生成，选择Export Dataset导出。

![image-20230713141743467](C:\Users\leesh\AppData\Roaming\Typora\typora-user-images\image-20230713141743467.png)

这里提供两种方式，一种直接下载，一种用python代码下载（需要pip install roboflow）。代码下载在JupyterLab中使用更方便。

![image-20230713144356838](C:\Users\leesh\AppData\Roaming\Typora\typora-user-images\image-20230713144356838.png)

下载压缩包后解压，文件夹中有这些文件。

## 训练

YOLOv8提供了两种使用模型的方法，一种通过CLI，另一种通过Python代码。这里我使用的Python代码的方法。

YOLOv8可以进行检测（detection），分割（segmentation），分类（classification）。这里只讲detection。

首先需要导入YOLO

```python
from ultralytics import YOLO
```

选择一个预训练的模型导入

```python
model = YOLO('yolov8n.pt')
```

这里有多个不同的预训练模型，里面的参数不同。模型的区别具体表现在性能和精度上。性能更好的精度就较差。下表为各个模型的性能表，可以根据需求选择。

![image-20230713143227506](C:\Users\leesh\AppData\Roaming\Typora\typora-user-images\image-20230713143227506.png)

导入模型后，开始训练

```python
model.train(data="data.yaml", epochs=100)
```

这里有很多参数，具体可以在文档网站中查看，这里我只列举几个。

[Train - Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/modes/train/#arguments)

|   Key    | Value  |                         Description                          |
| :------: | :----: | :----------------------------------------------------------: |
|   data   |  None  |             path to data file, i.e. coco128.yaml             |
|  epochs  |  100   |                number of epochs to train for                 |
| `batch`  |  `16`  |        number of images per batch (-1 for AutoBatch)         |
| `device` | `None` | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu |
|  `save`  | `True` |          save train checkpoints and predict results          |

如果是选择用自己的数据，则直接把data设置成从roboflow下载的文件夹里的'data.yaml'。训练设备默认使用CPU，要选择GPU训练，必须安装了cuda以及GPU版本的pytorch。在这里设置device = 0即可使用GPU训练。如果多GPU训练，设置例如device = [0,1,2]。

训练完成后，训练模型保存在 /runs/detect 目录中，名字为train。train中保存了本次训练的训练过程图示，以及各类数据。你可以在这里看到一些验证结果。

![val_batch2_pred](F:\yolov8\ultralytics\runs\detect\defult\val_batch2_pred.jpg)

其中，weights文件夹中的两个pt文件，best.pt 和 last.pt，是训练模型本体。后续验证或预测需要用到。

## 验证

这里的'best.pt'要用刚才训练完成的模型本体。

```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.val()
```

## 预测

预测的输入文件可以是图片、URL、视频、截屏等等。

| Source      | Argument                                   | Type                                  | Notes                                                        |
| ----------- | ------------------------------------------ | ------------------------------------- | ------------------------------------------------------------ |
| image       | `'image.jpg'`                              | `str` or `Path`                       | Single image file.                                           |
| URL         | `'https://ultralytics.com/images/bus.jpg'` | `str`                                 | URL to an image.                                             |
| screenshot  | `'screen'`                                 | `str`                                 | Capture a screenshot.                                        |
| PIL         | `Image.open('im.jpg')`                     | `PIL.Image`                           | HWC format with RGB channels.                                |
| OpenCV      | `cv2.imread('im.jpg')`                     | `np.ndarray` of `uint8 (0-255)`       | HWC format with BGR channels.                                |
| numpy       | `np.zeros((640,1280,3))`                   | `np.ndarray` of `uint8 (0-255)`       | HWC format with BGR channels.                                |
| torch       | `torch.zeros(16,3,320,640)`                | `torch.Tensor` of `float32 (0.0-1.0)` | BCHW format with RGB channels.                               |
| CSV         | `'sources.csv'`                            | `str` or `Path`                       | CSV file containing paths to images, videos, or directories. |
| video ✅     | `'video.mp4'`                              | `str` or `Path`                       | Video file in formats like MP4, AVI, etc.                    |
| directory ✅ | `'path/'`                                  | `str` or `Path`                       | Path to a directory containing images or videos.             |
| glob ✅      | `'path/*.jpg'`                             | `str`                                 | Glob pattern to match multiple files. Use the `*` character as a wildcard. |
| YouTube ✅   | `'https://youtu.be/Zgi9g1ksQHc'`           | `str`                                 | URL to a YouTube video.                                      |
| stream ✅    | `'rtsp://example.com/media.mp4'`           | `str`                                 | URL for streaming protocols such as RTSP, RTMP, or an IP address. |

这里我只用了图片和视频。

在后面如果要将YOLOv8写到自己的识别程序里面，就只能用图片逐帧预测。（也许有更好的方法但我不会）

```python
from ultralytics import YOLO

# 选取训练的模型
model = YOLO('run/detect/train/weights/best.pt')

# 视频预测
source = 'test.mp4'
results = model(source)

# 图片预测
source = 'test.jpg'
results = model(source)# 这里设置save=True会保存一个predict文件到 runs/detect 中
```

预测结果

![image-20230713151246856](C:\Users\leesh\AppData\Roaming\Typora\typora-user-images\image-20230713151246856.png)

预测的具体信息

```python
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Class probabilities for classification outputs
```

这里直接导入到supervision更好用。在后面会用到。

---

# 视频检测

在后面把模型部署到视频检测程序的时候，还学到了很多有用的东西，在此做个记录。这个可以作为视频检测的一个学习笔记。

## 头文件

```python
import os
import supervision as sv
import cv2 as cv
from ultralytics import YOLO
```



## 视频输入

根据输入参数选择视频的输入方式。

```python
if t == "video":
    # 视频输入
    cap = cv.VideoCapture(d + "/Video/test.mp4")
    cap.set(cv.CAP_PROP_FPS, 30) # 设置参数
elif t == "camera":
    # 摄像头输入
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FPS, 30)
else:
    print("invalid input")
    return
```



### cv2.VideoCapture

```python
设置参数
**propId**：  *parameter*    *function* 
0:CV_CAP_PROP_POS_MSEC   视频文件的当前位置（毫秒）
1:CV_CAP_PROP_POS_FRAMES 下一个要解码/捕获的帧的基于0的索引
2:CV_CAP_PROP_POS_AVI_RATIO 视频文件的相对位置：0-影片开始，1-影片结束
3:CV_CAP_PROP_FRAME_WIDTH 视频流中帧的宽度。
4:CV_CAP_PROP_FRAME_HEIGHT 视频流中帧的高度
5:CV_CAP_PROP_FPS 帧速率
6:CV_CAP_PROP_FOURCC 编解码器的4字符代码
7:CV_CAP_PROP_FRAME_COUNT 视频文件中的帧数
8:CV_CAP_PROP_FORMAT   retrieve()返回的Mat对象的格式
9:CV_CAP_PROP_MODE    后端特定的值，指示当前捕获模式
10:CV_CAP_PROP_BRIGHTNESS 图像的亮度（仅适用于相机）
11:CV_CAP_PROP_CONTRAST  图像的对比度（仅适用于相机）
12:CV_CAP_PROP_SATURATION  图像的饱和度（仅适用于相机）
13:CV_CAP_PROP_HUE  图像的色调（仅适用于相机）
14:CV_CAP_PROP_GAIN  图像增益（仅适用于相机）
15:CV_CAP_PROP_EXPOSURE  曝光（仅适用于相机）
16:CV_CAP_PROP_CONVERT_RGB 指示图像是否应转换为RGB的布尔标志
17:CV_CAP_PROP_WHITE_BALANCE  白平衡 目前不支持
18:CV_CAP_PROP_RECTIFICATION 立体声摄像机的校正标志

例如：
cap=cv2.VideoCapture(1)
cap.set(3, 1920)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

```



## 载入模型

```python
d = os.getcwd()
model = YOLO(d + "/runs/detect/default/weights/best.pt")
```



## 视频写出（逐帧写入到视频文件）

```python
# 获取视频的宽和高（不获取宽和高，写出时对应错误会报错）
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# 视频写出
# 文件 格式 帧率 分辨率 彩色
writer = cv.VideoWriter(d + '/Video/output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (width, height), True)
```



## 设置supervision标记框

```python
# 标记框参数
    box_annotator = sv.BoxAnnotator(
        thickness=2,  # 框线宽度
        text_thickness=2,  # 文字粗细
        text_scale=1  # 文字大小
    )
```



## While 循环逐帧处理

### 视频切片为帧

```python
# 切片
ret, frame = cap.read()
```



### 末帧判断

这个不要忘记，否则最后一帧获取不到frame会导致后面调用frame的函数报错。

```python
# 判断是否结束
if not ret:
    break
```



### 模型预测

```python
# 模型预测，agnostic参数 True表示多个类一起计算nms，False表示按照不同的类分别进行计算nms
# CPU：device = 'cpu'，GPU：device = 0 / device = [0,1]
result = model(frame, agnostic_nms=True, device=0)[0]
```

agnostic_nms参数为非极大值抑制（Non-maximum Suppression (NMS)）的作用简单说就是模型检测出了很多框，我应该留哪些。

设备选择算力更高设备可以加快检测速度。



## supervision

先把预测结果存入supervision.Detections，在用detections给lables赋值，最后重新绘制frame。

```python
# 预测结果存入supervision
#  xyxy：boxes坐标array  mask  confidence：boxes对应物体置信度  class_id：boxes对应类编号  tracker_id
detections = sv.Detections.from_yolov8(result)

#model.model.names[id]，返回编号对应物品类的名称
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
```



## 写出视频以及输出视频

```python
# 写入帧 注掉可以提高速度
writer.write(frame)

# 输出
cv.imshow("yolov8", frame)
```



## 退出

```python
# 每帧时长 输入esc退出
if cv.waitKey(10) == 27:
    break
```



## 释放

```python
#释放
cap.release()
writer.release()
cv.destroyAllWindows()

return
```





## 完整代码

```python
import os
import supervision as sv
import cv2 as cv
from ultralytics import YOLO


def animal_recognizer(t):
    d = os.getcwd()

    if t == "video":
        # 视频输入
        cap = cv.VideoCapture(d + "/Video/test.mp4")
        cap.set(cv.CAP_PROP_FPS, 30)
    elif t == "camera":
        # 摄像头输入
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FPS, 30)
    else:
        print("invalid input")
        return

    # 载入训练完成的模型
    model = YOLO(d + "/runs/detect/default/weights/best.pt")

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

    while True:
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

        #model.model.names[id]，返回编号对应物品类的名称
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

    #释放
    cap.release()
    writer.release()
    cv.destroyAllWindows()

    return


if __name__ == '__main__':
    # animal_recognizer("video")
    animal_recognizer("camera")

   
```



# OpenVINO

[Convert and Optimize YOLOv8 with OpenVINO™](https://docs.openvino.ai/2023.0/notebooks/230-yolov8-optimization-with-output.html)

参照以上链接，对CPU运行提速近一倍。GPU状态下提速不太明显。

未使用OpenVINO：

​	CPU 帧率： 8.187860972310833

​	GPU 帧率：16.976812571966995

使用OpenVINO：

​	CPU 帧率： 15.951011950592923

​	GPU 帧率： 21.718895497964738

（同一场景下）

​	
