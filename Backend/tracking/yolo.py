import torch
import cv2

# Load YOLOv10 model
# model_path = 'path/to/yolov10.pt'  # Adjust the path to your model file
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
from ultralytics import YOLOv10

# model = YOLOv10.from_pretrained('jameslahm/yolov10s')
model = YOLOv10.from_pretrained('jameslahm/yolov10x')

def process_frame(frame):
    return predict_and_detect(model,frame)

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = chosen_model(img,conf=0.3)[0]
    count = 0
    for result in results:
        for box in result.boxes:
            if box.cls[0] == 0:
                count+=1
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    cv2.putText(img, f"Person:{count}",
                    (10,len(img)-10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return img, results