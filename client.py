import requests
import json

image='/server9/ptz/ultralytics-main/ultralytics/yolo/v8/detect/img_fire.png'
with open(image, 'rb') as img_file:
    image = img_file.read()
    img_file.close()
files = {'image': image} 
url = 'http://127.0.0.1:9876/predict'

rl = requests.post(url, files=files)
info = json.loads(rl.text)

print(info['boxes'])