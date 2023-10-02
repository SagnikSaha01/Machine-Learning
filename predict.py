import supervision as sv
from ultralytics import YOLO
import cv2
import argparse
import numpy as np


model = YOLO("C:\\Users\\someone\\Desktop\\yolov8\\runs\\detect\\train22\\weights\\last.pt")
#model = YOLO('./runs/classify/train14/weights/last.pt')  # load a custom model
#liveResults = model.predict(source = "0",show = True)

#results = model('./data/weather_dataset/train/sunrise/sunrise1.jpg')  # predict on an image
camera = cv2.VideoCapture(0)

box_annotations = sv.BoxAnnotator(thickness = 2,text_thickness = 2,text_scale = 1,)
while True:
    ret,frame = camera.read()
    result = model(frame)[0]
    detections = sv.Detections.from_yolov8(result)
 
    labels = [
        f"{model.model.names[class_id]} {confidence: 0.2f}"
        for _, confidence, class_id, _
        in detections      
        ]
    frame = box_annotations.annotate(scene= frame,detections = detections,labels = labels)
    cv2.imshow("yolov8",frame)
    key = cv2.waitKey(1)