#%%
import cv2
import sys
import time
import torch
import math
import cvzone
from sort import sort
import numpy as np
 
#%% 
# Set up tracker.
# Initialize SORT tracker

mot_tracker = sort.Sort(min_hits=5, max_age=20)

#%% Set up model
model_path = r".\Trained_models\yolov5s_results_transfer_640\weights\best.pt" 
model = torch.hub.load('ultralytics/yolov5', 'custom',path = model_path)

#%%
# Read video
# video = cv2.VideoCapture("videos/video.mp4")
video = cv2.VideoCapture(0) # for using CAM

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()
#%%
# Read first frame.
ok, frame = video.read()
if not ok:
    print ('Cannot read video file')
    sys.exit()
    

# Uncomment the line below to select a different bounding box
#bbox = cv2.selectROI(frame, False)
ids_list = []
track_bbs_ids = np.empty((1,9))
frame_rate = 10
prev=0
while True:
    # Start timer
    timer = time.time() 

    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break   

    #Detect object(every frame)
    #Feed image to the model
    if timer-prev>1./frame_rate:
        prev = time.time()
        output  =model(frame)
        results = output.pred[0]
        time_data = output.t
        print(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image" % time_data)
    # output = model(frame)
    # results = output.pred[0]
    # t_detect = time.time()
    # print(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image" % output.t)

    #Draw bounding box and conf level
    for r in results:
        if len(r)>0:
            r = r.cpu().numpy()
            x1, y1, x2, y2 =r[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = (x1,y1,w,h)
            # Confidence
            conf = math.ceil((r[4] * 100)) / 100
            # Class Name
            cls = int(r[5])

            cvzone.cornerRect(frame,(x1,y1,w,h))
        
            #Update tracker
    #         track_bbs_ids = mot_tracker.update(results)
    #         print(track_bbs_ids)
    # if len(results)==0:
    #     #Update with empty list
    #     track_bbs_ids = mot_tracker.update(np.empty((0, 5)))

    #  # Draw bounding boxes with IDs for tracked objects
    # for j in range(len(track_bbs_ids.tolist())):
    #     coords = track_bbs_ids.tolist()[j]
    #     x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

    #     # Get the ID of the object
    #     if coords[4] not in ids_list: #Zasto je ovde 8 bilo?
    #         ids_list.append(coords[4])  # Add new ID to list if not already present
    #     name_idx = ids_list.index(coords[4])  # Get the index of the ID in the list

    #     # Create label with class and ID
    #     name_id = 'ID : {}'.format(str(name_idx))
    #     name_cls = 'CLS : {}'.format(str(coords[4]))
    #     name = name_cls + ' ' + name_id

    #     color = [0, 0, 255]  # Red color
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw rectangle around object
    #     cv2.putText(frame, name, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # Add label

   
    fps = 1/(time.time() - timer)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    if cv2.waitKey(10) & 0xFF == ord('q'): # if press SPACE bar
        break

cv2.destroyAllWindows()
# %%
video.release()

# %%
