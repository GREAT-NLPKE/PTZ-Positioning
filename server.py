import os
import json
import math
import base64
from io import BytesIO
from PIL import Image
import cv2 as cv
import numpy as np
from flask import Flask,jsonify,request,session
from flask import render_template,make_response
from flask_cors import CORS

from ultralytics import YOLO

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY']=os.urandom(24)

model = YOLO('best.pt')

def pil2cv(img:Image):
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
def cv2bytes(img):
    _, img_encode = cv.imencode('.jpg', img)
    img_bytes = img_encode.tobytes()
    return img_bytes

class PTZ:
    def __init__(self) -> None:
        self.intrinsics = np.zeros((3, 3))
        self.extrinsics = np.zeros((4, 4))
        self.radius_earth = 6371000   # 千米
        
    def getExtrinsics(self,height,P,T):
        rotation_matrix = self.eulerAngles2rotationMat([-90+T,0,P],format="degree")
        self.extrinsics[:3, :3] = rotation_matrix
        self.extrinsics[2][3] = height
        self.extrinsics[3][3] = 1
        # print("外参:")
        # print(self.extrinsics)

    def getIntrinsics(self,Z,h,w,f,sensor_h=5.32,sensor_w=7.18):
        d_x =sensor_w/w
        d_y =sensor_h/h
        In = [
            [f*Z/d_x,    0,    w/2],
            [0    ,f*Z/d_y,    h/2],
            [0 , 0 , 1]
        ]
        self.intrinsics = np.array(In)
        # print("内参:")
        # print(self.intrinsics)
    
    def eulerAngles2rotationMat(self,theta, format='degree'):
    #theta = [a , b, c] a为绕x角度，b为绕y角度，c为绕z角度，遵循ZYX内旋法则
        if format == 'degree':
            theta = [i * math.pi / 180.0 for i in theta] # 角度转弧度

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def getxyz(self,u,v): # 得到图像中火点在世界坐标系下的3D坐标
        uv = np.array([[u],[v],[1]])
        K_inv = np.linalg.inv(self.intrinsics)
        tempResult = K_inv@uv
        # print("相机坐标系下坐标：")
        # print(tempResult)
        tempResultp = np.ones((4,1))
        tempResultp[:3] = tempResult
        result = self.extrinsics@tempResultp
        # print("point2世界坐标:")
        # print(result)
        return result.tolist()

    def getTargetOffset(self,x1,y1,z1,x2,y2,z2,initAngle): # 地面火点坐标
        x0 = z1*(x2-x1)/(z1-z2) + x1
        y0 = z1*(y2-y1)/(z1-z2) + y1
        n = np.sqrt(x0*x0+y0*y0)*np.cos((initAngle + np.degrees(np.arctan(x0/y0)))*np.pi/180)
        m = np.sqrt(x0*x0+y0*y0)*np.sin((initAngle + np.degrees(np.arctan(x0/y0)))*np.pi/180)
        return m,n

    def getr(self,latitude): # 获得 火点纬度处地球半径
        r =  np.cos(latitude * np.pi/180) * self.radius_earth
        return r
    
    def getResult(self,longtitude,latitude,initAngle,u,v):
        # print(u,v)
        point2 = self.getxyz(u,v)
        m,n = self.getTargetOffset(0,0,self.extrinsics[2][3].tolist(),
            point2[0][0],point2[1][0],point2[2][0],initAngle)
        print("偏移量：",m,n)
        r=self.getr(latitude)  # 火点处地球周长
        longtitude_t = longtitude + m*180/(r*np.pi)  # 相机经纬度+火点偏移量
        latitude_t = latitude + n*180/(self.radius_earth*np.pi )
        print(longtitude_t,latitude_t)
        return longtitude_t, latitude_t


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        img = request.files["image"].read()
        img = Image.open(BytesIO(img)) 
        img_cv = pil2cv(img=img)
        results = model(img)
        boxes = results[0].boxes.xyxy
        if len(boxes)>0:
            boxes = [int(box) for box in boxes[0]]
            x1,y1,x2,y2 = boxes
            cv.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
        else:
            boxes = []
        img_bytes = cv2bytes(img_cv)
        img_stream = base64.b64encode(img_bytes).decode()
        print('boxes:',boxes)
        return jsonify({'width':img.width,'height':img.height,'boxes':boxes,'output_image':img_stream})

@app.route('/locate',methods=['GET','POST'])
def locate():
    if request.method == 'GET':
        x1 = int(request.args.get('x1'))
        y1 = int(request.args.get('y1'))
        x2 = int(request.args.get('x2'))
        y2 = int(request.args.get('y2'))
        
        ptz = PTZ()
        ptz.getIntrinsics(float(request.args.get('para_z')), int(request.args.get('para_image_h')), int(request.args.get('para_image_w')), int(request.args.get('para_focal')))
        ptz.getExtrinsics(int(request.args.get('para_height')),float(request.args.get('para_p')),float(request.args.get('para_t')))

        lon,lat = ptz.getResult(float(request.args.get('para_longitude')), float(request.args.get('para_latitude')),float(request.args.get('para_initangle')),u=int((x1+x2)/2),v=int((y1+y2)/2))
        # predict_smoke_pos = [lat,lon]
        # print('预测火点告警的经纬度为:',lon,lat)
        return jsonify({'latitude':lat,'longitude':lon})
    
@app.route('/check',methods=['GET'])
def check():
    if request.method == 'GET':
        ptz = PTZ()
        print(model.device)
        return jsonify({'state':True})

@app.route('/')
def index():
    resp = make_response(render_template("index.html"))
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9876)
