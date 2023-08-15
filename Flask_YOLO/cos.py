import requests
import cv2 as cv
import numpy as np
import base64

url = "http://192.168.0.238:8000/img"

file_path = 'C:/Users/leesh/Desktop/animal_recognizer/Flask_YOLO/1.jpg'

with open(file_path, 'rb') as f:
    img = base64.b64encode(f.read()).decode()

image = []
image.append(img)
res = {"image": image}


r = requests.post(url, data=res)

result = r.content

print(result)
