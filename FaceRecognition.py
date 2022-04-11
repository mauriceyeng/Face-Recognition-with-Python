import numpy as np
import cv2
import os

os.chdir(r"D:\LPU\Year 3\6th sem\ML Project")

face_cascade=cv2.CascadeClassifier("harrcascade.xml")
eye_cascade=cv2.CascadeClassifier("cascadeeyes.xml")
smile_cascade=cv2.CascadeClassifier("cascadesmile.xml")


def detect_face(image):
    face_copy=image.copy()
    
    face_rects=face_cascade.detectMultiScale(face_copy,1.3,5)
    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_copy,(x,y),(x+w,y+h),(0,0,255),5)
        
    eye_rects=eye_cascade.detectMultiScale(face_copy,1.1,10)
    for(x,y,w,h) in eye_rects:
        cv2.rectangle(face_copy,(x,y),(x+w,y+h),(0,0,255),5)
    
    
    smile_rects=smile_cascade.detectMultiScale(face_copy,1.7,22)
    for(x,y,w,h) in smile_rects:
        cv2.rectangle(face_copy,(x,y),(x+w,y+h),(0,0,255),5)
        
    return face_copy
                      
                      
capture=cv2.VideoCapture(0)

while True:
    ret,frame=capture.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    output=detect_face(frame) 
    cv2.imshow("output",output)
    
    if cv2.waitKey(1) & 0xff==ord('a'):
        break
        
capture.release()
cv2.destroyAllWindows()