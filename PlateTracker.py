from ultralytics import YOLO
import cv2 as cv
import cvzone
import numpy as np
import math
from sort import *
import numpy as np
import pytesseract as pt
import re
import pandas as pd
import os
import keras_ocr
import warnings
warnings.filterwarnings('ignore')



model = YOLO('../yolo_weights/best.pt')

class_name = ['License Plate']

capture = cv.VideoCapture('1 short.mp4')

tracker = Sort(max_age=5,min_hits=2,iou_threshold=0.3)


crossedline1 = [1000,1000,1950,1000] # red
crossedline2 = [150,1000,890,1000] # green



def PlateCapture():
    while True:
        ret,frame = capture.read()
        result = model(frame,stream = True) # stream = True , use generator objects
        dets=np.empty((0, 5)) # tracker.update is intialized as an empty array that changes after different frames

        if not ret:
        # Break the loop if there are no more frames to read
            break

        for r in result: # r has tensors in it
            boxes = r.boxes
            for coord in boxes:
                xmin,ymin, xmax , ymax = coord.xyxy[0] # we can also use coord.xyxy
                xmin,ymin, xmax , ymax = int(xmin),int(ymin), int(xmax) , int(ymax)

                bbox = xmin,ymin, xmax-xmin , ymax-ymin # bbox is bounding box

                confidence = (math.ceil(coord.conf[0]*100))/100

                clsname = int(coord.cls[0])

                arr = np.array([xmin,ymin,xmax,ymax,confidence])
                dets = np.vstack((dets,arr)) # appending the changes vertically 

        trackres = tracker.update(dets)
        
        for t in trackres:
            xmin,ymin,xmax,ymax,id = t
            xmin,ymin,xmax,ymax,id = int(xmin),int(ymin),int(xmax),int(ymax),int(id)
            w,h = xmax - xmin , ymax - ymin

            midpointx , midpointy = (xmin+xmax)//2,(ymin+ymax)//2
            
            if (crossedline2[0] < midpointx < crossedline2[2]) and (crossedline2[1] - 15 < midpointy < crossedline2[3] + 15):
                roi1 = frame[ymin:ymax,xmin:xmax]
                
                cv.imwrite(f'LicensePlate/{id}.jpg',roi1)

            if (crossedline1[0] < midpointx < crossedline1[2]) and (crossedline1[1] - 15 < midpointy < crossedline1[3] + 15):
                roi2 = frame[ymin:ymax,xmin:xmax]
                cv.imwrite(f'LicensePlate/{id}.jpg',roi2)

        cv.namedWindow('image', cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty('image', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow('image',frame)
        cv.waitKey(1)

    capture.release()
    cv.destroyAllWindows()
