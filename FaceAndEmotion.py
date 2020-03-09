import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

#load model
model = model_from_json(open("fer9.json", "r").read())
#load weights
model.load_weights('fer9.h5')

## Loading the frontal face classifier to detect a face's front view
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#read video stream from webcam using VideoCapture
# 0 used because using the laptop's built-in camera
cam=cv2.VideoCapture(0)

#define and load the recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create();
recognizer.read('C:/Users/ghulam/Desktop/my project/face rec/recognizer/trainer5.yml') #loading the trained yml file
id=0
print('You can now detect and recognise faces and emotions :D') 


while True:
    ret,img=cam.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        ############   FACES #######################################
        id,conf=recognizer.predict(gray_img[y:y+h,x:x+w])
        if(id==1):
             id="leo"
             cv2.putText(img,str(id),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255));
        elif(id==2):
             id="Obama"
             cv2.putText(img,str(id),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255));
        elif(id==4):
             id="emanah"
             cv2.putText(img,str(id),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255));
       
        else:
             cv2.putText(img,"Unknown Face",(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255));
             #################################################
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
         
        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
    cv2.imshow('You can now detect & recognise faces and emotions :D ',img)


    if cv2.waitKey(1) == ord('x'):#wait until 'q' key is pressed
        break
###########
cam.release()
cv2.destroyAllWindows