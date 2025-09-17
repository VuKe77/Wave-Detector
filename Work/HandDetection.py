#%%
import numpy as np
import cv2
import cvzone
import math
import torch
import time
#%% Load model
model_path = r".\Trained_models\yolov5s_results_transfer_640\weights\best.pt"
#model = YOLO(model_path,augment=True)   
model = torch.hub.load('ultralytics/yolov5', 'custom',path = model_path)


#%% Inference
cap = cv2.VideoCapture(0)  # For Video
 
classNames = ['fist','palm','no_gesture']
 
frame_rate = 10
prev = 0
while True:

    time_elapsed = time.time()-prev

    #Read image from camera
    success, img = cap.read()

    #Feed image to the model
    if time_elapsed>1./frame_rate:
        prev = time.time()
        output  =model(img)
        results = output.pred[0]
        time_data = output.t
        print(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image" % time_data)

        #Draw bounding box and conf level
        for r in results:
            if len(r)>0:
                r = r.cpu().numpy()
                x1, y1, x2, y2 =r[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                # Confidence
                conf = math.ceil((r[4] * 100)) / 100
                # Class Name
                cls = int(r[5])

                cvzone.cornerRect(img,(x1,y1,w,h))
                cvzone.putTextRect(img, f'{classNames[cls]}:{conf}', (max(0, x1), max(35, y1)),
                            scale=2, thickness=3, offset=10)
        #Show image
        cv2.imshow("Camera",img)
    if cv2.waitKey(5) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 



# %%
