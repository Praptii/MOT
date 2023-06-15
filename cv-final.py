# Author: Prapti Thapaliya
# Villanova University
# Date Created: 12/08/2022
#
# Multi Object Detector

import torch
from torch import hub 
import math
from scipy.spatial import distance
import cv2
import numpy as np
import pandas as pd
from tkinter.filedialog import askopenfilename

model = torch.hub.load( 'ultralytics/yolov5', 'yolov5s', pretrained=True)
closest_distance_threshold = 50
tracker_kill_count = 30

class person_tracker:
    def __init__(self ,x1, y1, x2, y2):
        self.last_updated = 0
        self.update_points(x1, y1, x2, y2)
        self.active = True
    
    def set_midpoint(self):
        self.x = (self.x1 + self.x2) / 2
        self.y = (self.y1 + self.y2) / 2
        
    def update_points(self,x1, y1, x2, y2):
        global frame_count
        
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.set_midpoint()
        self.last_updated = frame_count
    
    def check_inactivity(self):
        global frame_count
        global tracker_kill_count
        
        if (frame_count - self.last_updated ) > tracker_kill_count:
            self.active = False

def kill_inactive_trackers(trackers):
    for idx in range(len(trackers)):
        trackers[idx].check_inactivity()
                    
def check_closest_distance(trackers, detection):
    global closest_distance_threshold
    
    x1,y1,x2,y2 = detection[0], detection[1], detection[2], detection[3]
    xd, yd = (x1 + x2) / 2 , (y1 + y2) / 2
    
    minimum = 1000000
    min_index = -1
    
    ids = len(trackers)

  
    for i in range(ids):
        
        if trackers[i].active:
            dist = distance.euclidean((xd, yd), (trackers[i].x, trackers[i].y))

            if dist < minimum and dist < closest_distance_threshold:
                minimum = dist
                min_index = i
            
    return min_index 

def xyxy2xywh(xyxy):
    """
    Convert bounding box coordinates from (xmin, ymin, xmax, ymax) format to (xmin, ymin, width, height).

    Args:
        xyxy (numpy.ndarray):

    Returns:
        numpy.ndarray: Bounding box coordinates (xmin, ymin, width, height).

    """

    (left, top, right, bottom) = xyxy
    width = right - left + 1
    height = bottom - top + 1
    return [int(left), int(top), int(width), int(height)]

font = cv2.FONT_HERSHEY_SIMPLEX
filename = askopenfilename()
cap = cv2.VideoCapture(filename)
colors = [tuple(np.random.randint(256, size=3)) for x in range(1000)]

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
result = cv2.VideoWriter('result.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)


trackers = {}
tracker_id = 0
frame_count = 0

ret, frame = cap.read()

records = []

while True:
    key = cv2.waitKey(1) 
    
    if key == ord('p'):
        break
    
    output = model(frame)
    
    detections = output.xyxy[0]

    for detection in detections:
        
        # Only detect People. Class value for people is 0
        if int(detection[5].cpu().numpy()) == 0:
            
            x1,y1,x2,y2 = detection[0].cpu().numpy(), detection[1].cpu().numpy(), detection[2].cpu().numpy(), detection[3].cpu().numpy()
            
            idx = check_closest_distance(trackers, (x1,y1,x2, y2))
            
            if idx == -1:
                trackers[tracker_id] = person_tracker(x1, y1, x2, y2)
                tracker_id += 1
            else:
                trackers[idx].update_points(x1, y1, x2, y2)
    
    kill_inactive_trackers(trackers)
    
    for t_id in range(len(trackers)):
        
        if trackers[t_id].active:
            color = colors[t_id]
            color_x = (int(color[0]), int(color[1]), int(color[2]))
            cv2.rectangle(frame, (int(trackers[t_id].x1), int(trackers[t_id].y1)), (int(trackers[t_id].x2), int(trackers[t_id].y2)), color_x, 2)
            cv2.putText(frame, str(t_id)  , (int(trackers[t_id].x) - 10, int(trackers[t_id].y)), font, 1,  color_x, 2)
            
            xywh = xyxy2xywh((trackers[t_id].x1, trackers[t_id].y1,trackers[t_id].x2, trackers[t_id].y2))
            records.append((frame_count, t_id, xywh[0], xywh[1], xywh[2], xywh[3], 1, -1, -1, -1 ))
        
    # cv2_imshow(frame)
    result.write(frame)

    frame_count += 1
    ret, frame = cap.read()
    
    

    if ret == False:
        break
        
cap.release()
result.release()
cv2.destroyAllWindows()

# create detection csv
det_file = pd.DataFrame(records, columns=['frame_no','track_id', 'x','y','w','h', 'n1', 'n2', 'n3', 'n4'])
det_file.to_csv('det_file.csv')

