from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")


results = model.train(data="config.yaml",epochs = 15,save = True,batch = 2)
