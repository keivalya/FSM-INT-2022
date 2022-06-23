import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp15/weights/best.pt', force_reload=True)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow("PCB Fault Detection", np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()