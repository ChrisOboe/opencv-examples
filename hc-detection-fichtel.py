#!/usr/bin/python

import cv2

cascade_face = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)
img_fichtel = cv2.imread('/home/chris/lars2.png')

while(True):
    ok, frame = vid.read()
    if (ok == False):
        print("Couldn't open video")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detected_faces = cascade_face.detectMultiScale(frame_gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        # fichtelate
        img_fichtel_sized = cv2.resize(img_fichtel, (w, h))
        frame[y:y+h, x:x+w] = img_fichtel_sized


    cv2.imshow('vid', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
