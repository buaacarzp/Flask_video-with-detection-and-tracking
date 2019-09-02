#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response
from flask import request, Flask,send_file,jsonify
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
import base64
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,
    help="path to input video file")
ap.add_argument("-l", "--label", required=True,
    help="class label we are interested in detecting + tracking")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
    print("1")
else:
#    from camera import Camera  # zp change here
#    from camera_opencv import Camera
    print("2")

from base_camera import BaseCamera

class Camera(BaseCamera):
    video_source = 0
    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        tracker=None
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            global startX, startY,endX,endY
            # read current frame
            _, img = camera.read()
            frame=img
            if frame is None:
                break
            frame = imutils.resize(frame, width=800)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#
            if tracker is None:
                tracker = dlib.correlation_tracker()
#                global startX, startY,endX,endY
                print("the xyxyis \n",startX, startY,endX,endY)
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                tracker.start_track(rgb, rect)
            else:
                tracker.update(rgb)#update() for子序列
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

        # draw the bounding box from the correlation object tracker
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(frame, "tracking", (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            yield cv2.imencode('.jpg', frame)[1].tobytes()

def imageToStr(image):
    with open(image,'rb') as f:
        image_byte=base64.b64encode(f.read())
        print(type(image_byte))
    image_str=image_byte.decode('ascii') #byte类型转换为str
    print(type(image_str))
    return image_str

app = Flask(__name__)

@app.route('/1',methods=['POST'])
def detection():
    '''return the detection image'''
    print("[INFO] starting video stream...")
#    time.sleep(3)
    vs = cv2.VideoCapture(0)
    #参数传入数字时，代表打开摄像头，传入本地视频路径时，表示播放本地视频
    global tracker
    global s
    tracker=None
    label = ""
    _,frame = vs.read()#fps:开始记录第一帧
    frame = imutils.resize(frame, width=800)
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5,False)
#    blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5) 
    net.setInput(blob)
    detections = net.forward()
    print("detections is :\n",detections)
    print("type detections is:",type(detections))
    rezp=detections.reshape(detections.shape[2],7)
    data_cs = rezp[(-(rezp[:, 2])).argsort()] 
    print("降序排列：\n",data_cs)
    detections=data_cs
    
    '''
    检测模块
    '''
    s=[]
    print(type(s))
    # loop over the detections
    for i in np.arange(0, detections.shape[0]):#画出第一帧的目标检测结果
        confidence = detections[i, 2]
        label_1="box{}".format(i)
        if confidence > 0.8 :
            idx = int(detections[i, 1])
        
#        box = detections[0, 0, i, 3:7] * np.array([w*(2), h, w*(3/4), h*(3/4)])
            box = detections[i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
        
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
        
            y = startY - 15 if startY - 15 > 15 else startY + 15
        
            print("the startX, y is:\n",(startX, startY), (endX, endY))
        
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            cv2.putText(frame, label_1, (startX+30, y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            s.append([startX,startY,endX, endY])
            vs.release()
    cv2.imwrite("./cv2img.jpg",frame)
    print("s1:",s)
    print("Now have {} boxes!".format(len(s)))
    image="./cv2img.jpg"
    image1=imageToStr(image)
    dict1=jsonify({'img':image1,'s':len(s)})
    return dict1
@app.route("/4", methods=['POST'])
def get_4():
    global tracker
    global startX,startY,endX,endY
    global s#s是否能传进来，之前是由检测的结果传入跟踪，现在改为 c#选择的结果传入跟踪
    username = request.form.get('item')
#    username="box0"
    print("/4 ： 传入进来的s为",s)
    print("username type is:",type(username))
    print("yuanyuan had transform is",username)
    box_num=username.split("box",1)[1]
    print("box_num is:",box_num)
    startX,startY,endX,endY=s[int(box_num)]
    return "I have received the box you choosed"

@app.route('/2')#跟踪模块
def index():
    """Video streaming home page."""
    print("3")
    return render_template('index.html')


def gen(camera):#实际上是类作为参数传递，也就是实例化，camera=Camera()
    """Video streaming generator function."""
    print("4")
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    print("5")
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
