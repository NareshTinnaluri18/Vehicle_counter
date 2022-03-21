#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[7]:


cap = cv2.VideoCapture(r'C:\Users\Chocky Naresh\Desktop\Computer Vision\vehicle_counter_new\Cars Moving On Road Stock Footage - Free Download.mp4')

min_width_rect = 80
min_height_rect = 80

count_line_position = 550

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1= int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
offset = 6
counter = 0


while True:
    success, frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    counterShape,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25,count_line_position),(1600, count_line_position),(255,127,0),3)
    
    
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame,center,4,(0,0,255),-1)
        
        
        for(x,y) in detect:
            if y <(count_line_position+offset) and  y <(count_line_position+offset):
                counter+=1
            cv2.line(frame, (25,count_line_position),(1600, count_line_position),(0,127,255),3)
            detect.remove((x,y))
            print("vehicle counter:"+str(counter))
            
    cv2.putText(frame,"vehicle_counter:"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
    
           
    
    cv2.imshow('Video_original',frame)
    if cv2.waitKey(1)==13:
        break
cv2.destroyAllWindows()
cap.release()


# In[ ]:




