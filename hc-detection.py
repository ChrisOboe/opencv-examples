#!/usr/bin/python

import cv2

cascade_face = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_eye.xml')
#cascade_smile = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_smile.xml')

img_org = cv2.imread('HERE SHOULD BE YOR IMAGE')
img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

cv2.imshow('img', img_org)
cv2.waitKey(0)

detected_faces = cascade_face.detectMultiScale(img_gray)

for (x, y, w, h) in detected_faces:
    cv2.rectangle(img_org, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # get this face
    img_face = img_gray[y:y+h, x:x+w]
    # look for eyes
    detected_eyes= cascade_eye.detectMultiScale(img_face)
    eye_count = 0
    for (ex, ey, ew, eh) in detected_eyes:
        eye_count+=1
        cv2.rectangle(img_org, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 2)

#    detected_smile = cascade_smile.detectMultiScale(img_face)[0]
#    cv2.rectangle(img_org, (x+detected_smile[0], y+detected_smile[1]), (x+detected_smile[0]+detected_smile[2], y+detected_smile[1]+detected_smile[3]), (0,0,255), 2)

cv2.imshow('img', img_org)
cv2.waitKey(0)
cv2.destroyAllWindows()
