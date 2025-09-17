#%%
import cv2
import sys
import time
import torch
import math
import cvzone
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#%% 


# Set up tracker.
# Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[-1]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    # elif tracker_type == 'GOTURN':
    #     tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
#%% Set up model
model_path = r".\Trained_models\yolov5s_results_640_exp3\weights\best.pt" 
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
    
# Define an initial bounding box
bbox1 = (250, 100, 100, 100)
# Uncomment the line below to select a different bounding box
#bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox1)
t_detect = 0
while True:
    # Start timer
    timer = time.time() 

    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break   

    #Detect object(every second)
    if timer-t_detect>1:
        output = model(frame)
        results = output.pred[0]
        t_detect = time.time()
        print(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image" % output.t)

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
            
            ok = tracker.init(frame, bbox)

    else:

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = 1/(time.time() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
        break

cv2.destroyAllWindows()
# %%
video.release()
