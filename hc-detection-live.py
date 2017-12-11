#!/usr/bin/python

import cv2

cascade_face = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_eye.xml')
#cascade_smile = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_smile.xml')

vid = cv2.VideoCapture(0)

while(True):
    ok, frame = vid.read()
    if (ok == False):
        print("Couldn't open video")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detected_faces = cascade_face.detectMultiScale(frame_gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # get this face
        frame_face = frame_gray[y:y+h, x:x+w]
        # look for eyes
        detected_eyes= cascade_eye.detectMultiScale(frame_face)
        for (ex, ey, ew, eh) in detected_eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 2)

    cv2.imshow('vid', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
