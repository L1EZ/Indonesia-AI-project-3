import torch
import cv2

# Load YOLOv10 model
# model_path = 'path/to/yolov10.pt'  # Adjust the path to your model file
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
from ultralytics import YOLO
import time
import random
import cv2
import numpy as np
import time
import os
traffic = YOLO("tracking/best.pt",task="classify")
model = YOLO("tracking/yolov8l-seg.pt",task="segment")
print(os.getcwd())

def process_frame(frame):
    return predict_and_detect(model,frame)

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    conf = 0.5
    baseImage = img.copy()
    results = model.predict(img, conf=conf)
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    ct = 1
    current = time.time()
    for result in results:
        for mask, box,cnf,label in zip(result.masks.xy, result.boxes,result.boxes.conf,result.boxes.cls):
            points = np.int32([mask])
            overlay = img.copy()
            cv2.polylines(overlay, points, True, (255, 0, 0), 1)
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.fillPoly(overlay, points, colors[color_number])
            label_text = f'{yolo_classes[int(label)]} {cnf:.2f}'
            xys = np.int32(box.xyxy.cpu()[0]).reshape(2,2)
            x = xys[0][0]
            y = xys[0][1]
            x2 = xys[1][0]
            y2 = xys[1][1]
            overlay = cv2.rectangle(overlay, xys[0],xys[1], (0, 0, 255), 2)
            img = cv2.addWeighted(overlay,0.5,img, 0.5,0)
            offset = 50
            if int(label) == 9:
                maxy = len(baseImage)
                maxx = len(baseImage[0])
                xl =x-offset
                xr = x2+offset
                yl =y-offset
                yr = y2+offset
                if yl <0:
                    yl = 0
                if yr >= maxy:
                    yr = maxy-1
                if xl < 0:
                    xl = 0
                if xr >= maxx:
                    xr = maxx-1
                imgCopy = baseImage[yl:yr,xl:xr]
                cv2.imwrite(f"crop{ct}.jpg",imgCopy)
                ct+=1
                res = traffic(imgCopy)
                print(res[0].tojson())
                if len(res[0].boxes) >= 1:
                    print("classify")
                    print(res[0])
                    trafficLight = traffic.names[int(res[0].boxes[0].cls)]
                    label_text = f'{trafficLight} {cnf:.2f}'
                
            cv2.putText(img, label_text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
    print(time.time()-current)
    cv2.imwrite("test.jpg", img)
    return img, results