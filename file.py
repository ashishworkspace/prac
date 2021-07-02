#!/usr/bin/python3

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import smtplib
from skimage import io
import cgi 		

print("content-type: text/html")              
print("Access-Control-Allow-Origin: *")  
print()                                

form = cgi.FieldStorage()	      
links3  = form.getvalue("var")	              

img = io.imread(links3)
#img = cv2.imread('https://mybucketmodel.s3.ap-northeast-3.amazonaws.com/image3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour,10, True)
    if len(approx) == 4:
        location = approx
        break
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,500, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)
(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
if result != []:
    print(result[0][1])
else: 
    print("Not the Clear Image")
