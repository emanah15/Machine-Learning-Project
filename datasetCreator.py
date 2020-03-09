# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:38:22 2020

@author: ghulam
"""

import cv2

def generate_dataset(img, user_id, img_id):
    cv2.imwrite("dataSet/user."+str(user_id)+"."+str(img_id)+".jpg", img)

def draw_square(img, classifier, scaleFactor, minNeighbours, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []
    
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

# Method to detect the features
def detect(img, faceCascade, img_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0)}
    coords = draw_square(img, faceCascade, 1.15, 10, color['red'], "Face")
    
    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        
        user_id=3 # new number for collecting samples of a new person e.g entering 4 for a new person
        generate_dataset(roi_img, user_id, img_id)
    return img


# Loading classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
   

#read video stream from webcam
video_capture = cv2.VideoCapture(0)

img_id=0
while True:
    _, img = video_capture.read()
        # Call method we defined above
    img = detect(img, faceCascade, img_id)
    cv2.imshow("face detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q') or img_id==100:
        break
    
video_capture.release()
cv2.destroyAllWindows()
print('collecting samples complete!!!!')