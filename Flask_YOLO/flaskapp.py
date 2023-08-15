from flask import Flask, render_template, Response, jsonify, request, session,make_response
import cv2 as cv
from YOLO_Video import video_detection
from gevent import pywsgi
from YOLO_IMG import img_detection
import numpy as np
import base64

app = Flask(__name__)

# app.config['SECRET_KEY'] = 'tuffy'
# app.config['host'] = '127.0.2.1'


def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)

    for detection_ in yolo_output:
        ref, buffer = cv.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/doing')
def hello_doing():
    name = request.values.get('name')
    return 'hello {}'.format(name)

@app.route('/video')
def video():

    return Response(generate_frames(path_x="C:/Users/leesh/Desktop/animal_recognizer/Video/test.mp4"),content_type="image/jpeg")

@app.route('/img',methods=['POST'])
def img():
    img = base64.b64decode(str(request.form['image']))
    image_data = np.fromstring(img,np.uint8)
    image_data = cv.imdecode(image_data,cv.IMREAD_COLOR)



    return img_detection(image_data)




if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8000)
    # server = pywsgi.WSGIServer(('192.168.0.238', 8080), app)
    # server.serve_forever()
