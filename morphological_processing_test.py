#!/usr/bin/env python

import cv2 as cv
import sys
import numpy as np
from PIL import Image

image_path = sys.argv[1]
image = cv.imread(image_path)
image_one = cv.cvtColor(image, cv.COLOR_BGR2XYZ)
image_two = cv.cvtColor(image_one, cv.COLOR_BGR2GRAY)

threshold = 128
num_of_rows = len(image)
num_of_cols = len(image[0])

cv.imwrite('images/temp_XYZgray.jpg', image_two)

original_img = cv.imread("images/dhokHussu.jpg")


clahe = cv.createCLAHE(clipLimit=40.0, tileGridSize=(10, 10))
image_four = clahe.apply(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
cv.imwrite('images/sharpened.jpg', image_four)

kernel = np.ones((5,5),np.uint8)
image_five = cv.dilate(image_four, kernel, iterations = 1)
cv.imwrite('images/dilation.jpg', image_five)
image_six = cv.morphologyEx(image_four, cv.MORPH_OPEN, kernel)
cv.imwrite('images/opening.jpg', image_six)
image_seven = cv.morphologyEx(image_four, cv.MORPH_CLOSE, kernel)
cv.imwrite('images/closing.jpg',image_seven)

ret,thr = cv.threshold(image_seven,50,255,cv.THRESH_TRIANGLE,None)
im2, contours, hierarchy = cv.findContours(thr,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cv.imwrite('images/im2.jpg', im2)
#image_three = cv.drawContours(image, contours, -1, (0,255,0), 3)
#print type(contours)
#print len(image)

mask = np.zeros_like(original_img) # Create mask where white is what we want, black otherwise
cv.drawContours(mask, contours, -1, (255,255,255), -1) # Draw filled contour in mask
out = np.zeros_like(original_img) # Extract out the object and place into output image
out[mask == 255] = original_img[mask == 255]
cv.imwrite('images/temp_con_originl.jpg',out)
# Show the output image
#cv.imshow('Output', out)
#cv.waitKey(0)

#image_three = cv.GaussianBlur(out, (5,5), 0)
#image_four = cv.addWeighted(image_three, 1.5, out, -0.5, 0)
#cv.imwrite('images/sharpened.jpg', image_four)






