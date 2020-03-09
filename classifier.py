# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:08:18 2020

@author: ghulam
"""
import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='C:/Users/ghulam/Desktop/my project/face rec/dataSet'

def getImagesWithID(path):
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
   
    
	faces=[]
	IDs=[]
	for imagePath in imagePaths:
		faceImg=Image.open(imagePath).convert('L');
		faceNp=np.array(faceImg,'uint8')
		ID=int(os.path.split(imagePath)[-1].split('.')[1])
		faces.append(faceNp)
		#print ID
		IDs.append(ID)
		cv2.imshow('training',faceNp)
		cv2.waitKey(10)
	return IDs,faces

Ids,faces=getImagesWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('recognizer/trainer5.yml')
cv2.destroyAllWindows()
print("Training is Complete")