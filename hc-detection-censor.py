#!/usr/bin/python

import cv2

cascade_face = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

while(True):
    ok, frame = vid.read()
    if (ok == False):
        print("Couldn't open video")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detected_faces = cascade_face.detectMultiScale(frame_gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        # get this face
        frame_face = frame[y:y+h, x:x+w]
        # pixelate
        frame_face_small = cv2.resize(frame_face, (0,0), fx=0.0625, fy=0.0625, interpolation=cv2.INTER_NEAREST)
        frame_face_pixelated = cv2.resize(frame_face_small, (0,0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
        frame[y:y+frame_face_pixelated.shape[0], x:x+frame_face_pixelated.shape[1]] = frame_face_pixelated


    cv2.imshow('vid', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
