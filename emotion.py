import cv2
import mediapipe as mp
import numpy as np
import time
ptime=0
cap=cv2.VideoCapture(0)
mpDraw=mp.solutions.drawing_utils
mpface=mp.solutions.face_mesh
facemesh=mpface.FaceMesh(max_num_faces=1)
drawsprc=mpDraw.DrawingSpec(thickness=1,circle_radius=1)
hid=0
while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=facemesh.process(gray)
    if results.multi_face_landmarks:
        for facelens in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame,facelens,mpface.FACEMESH_CONTOURS,drawsprc,drawsprc)
            for l in facelens.landmark:
                h,w,c=frame.shape
                x,y=int(l.x*w),int(l.y*h)
                print(x,y)
                lower = np.array([200, 200, 200])
                upper = np.array([255, 255, 255])
                thresh = cv2.inRange(frame, lower, upper)
                
               
    hid+=1
    cv2.putText(frame,str(hid),(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)

    cv2.imshow("frame",frame)
    cv2.imwrite("data/"+'sad.'+str(hid)+'.png',thresh) 
    if cv2.waitKey(1)==ord("a") or int(hid)==300:
        break

cap.release()
cv2.destroyAllWindows()