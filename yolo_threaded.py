import os
import sys
import argparse
import glob
import time
import queue
import threading

import cv2
import numpy as np
from ultralytics import YOLO

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']



# Pick a tracker type (KCF, CSRT, MOSSE, etc.)
def create_tracker():
    params = cv2.TrackerNano_Params()
    params.backbone='nanotrack_backbone.onnx'
    params.neckhead='nanotrack_head.onnx'
    return cv2.TrackerNano_create(params)


cap = cv2.VideoCapture(0)

trackers = []
frame_count = 0
detect_interval = 5  # Run YOLO every n frames
track_id = 0

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
t_start = time.perf_counter()
bad_trackers=False

frame_queue = queue.Queue(maxsize=1)
trackers_queue = queue.Queue(maxsize=1)
tracker_data_queue = queue.Queue(maxsize=1)

def yolo_worker():
    while True:
        frame = frame_queue.get()
        results = model(frame, verbose=False)
        
        trackers = []
        tracker_data = {}
        bad_trackers=False
        
        object_count = 0
        track_id=0

        for detection in results[0].boxes:
            conf = detection.conf.item()
            if conf>0.5:
                box = detection.xyxy.cpu().numpy()
                xyxy_tensor = detection.xyxy.cpu() # Detections in Tensor format in CPU memory
                xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
                x1,y1,x2,y2 = xyxy.astype(int)
                
                w, h = x2-x1, y2-y1
                x1 = x1+1
                y1 = y1+1
                print(x1,x2,y1,y2)
                tracker=create_tracker()
                tracker.init(frame, (x1,y1,w,h))
                
                
                trackers.append((tracker, track_id))
                xmin, ymin, xmax, ymax  = x1-1, y1-1, x2-1, y2-1 
                
                classidx = int(detection.cls.item())
                classname = labels[classidx]

                color = bbox_colors[classidx % 10]
                
                object_count = object_count + 1

                tracker_data[track_id]={'color':color, 'classname':classname, 'conf':conf}

                track_id+=1
        
        if not trackers_queue.full():
            trackers_queue.put(trackers)
        if not tracker_data_queue.full():
            tracker_data_queue.put(tracker_data)

threading.Thread(target=yolo_worker, daemon=True).start()



while True:
    print(t_start - time.perf_counter())
    t_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % detect_interval == 0 or bad_trackers:
        if not frame_queue.full():
            frame_queue.put(frame.copy())
        
    if not trackers_queue.empty():
        trackers = trackers_queue.get()
    if not tracker_data_queue.empty():
        tracker_data = tracker_data_queue.get()

    if True: # remove and unindent if not needed
        bad_trackers=False
        # Update trackers
        new_trackers = []
        for tracker,track_id in trackers:
            success, bbox = tracker.update(frame)
            if success:
                
                x,y,w,h = [int(v) for v in bbox]
                
                xmin = x
                xmax = x+w
                ymin = y
                ymax = y+h
                
                color, classname, conf = tracker_data[track_id]['color'],tracker_data[track_id]['classname'],tracker_data[track_id]['conf']


                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text
                
                

                
                new_trackers.append((tracker,track_id))
                trackers = new_trackers
                
            else:
                bad_trackers=True
                print('bad')


    cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    

    cv2.imshow("YOLO + Tracker", frame)
    
    t_stop = time.perf_counter()
    time_passed = (t_stop - t_start)

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(time_passed)
    else:
        frame_rate_buffer.append(time_passed)

    # Calculate average FPS for past frames
    avg_frame_rate = 1/np.mean(frame_rate_buffer)
    

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
