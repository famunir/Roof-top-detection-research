#!/usr/bin/env python

import cv2 as cv
import sys
import numpy as np
from PIL import Image

image_path = sys.argv[1]
image = cv.imread(image_path)
image_eight = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_one = cv.cvtColor(image, cv.COLOR_BGR2XYZ)
image_two = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
image_three = cv.cvtColor(image, cv.COLOR_BGR2HSV)
image_four = cv.cvtColor(image, cv.COLOR_BGR2HLS)
image_five = cv.cvtColor(image, cv.COLOR_BGR2Lab)
image_six = cv.cvtColor(image, cv.COLOR_BGR2Luv)
#cv.imwrite('images/temp.jpg', image)

threshold = 128
num_of_rows = len(image)
num_of_cols = len(image[0])
#blank_image = np.zeros((num_of_rows, num_of_cols, 1), np.uint8)
#for ii in range(2, num_of_rows-1):
#	for jj in range(2,num_of_cols-1):
		#if image[ii][jj] >= threshold and blank_image[ii][jj] == 0:
			#blank_image[ii][jj] = image[ii][jj]
#		window = np.array([ [image[ii-1][jj-1], image[ii-1][jj], image[ii-1][jj+1]],
#			 [image[ii][jj-1], image[ii][jj], image[ii][jj+1]],
#			 [image[ii+1][jj-1], image[ii+1][jj], image[ii+1][jj+1]] ])
#		image[ii,jj] = np.median(window)

image_seven = cv.Canny(image_one, 50, 150)
image_eight = cv.Canny(image_eight, 100, 200)
image_nine = cv.Canny(image, 100, 200)
image_ten = cv.Canny(image_three, 100,200)
image_eleven = cv.Canny(image_four, 100,200)
# cv.imwrite('images/temp01.jpg', image)
cv.imwrite('images/tempXYZ.jpg', image_one)
cv.imwrite('images/tempCrCb.jpg', image_two)
cv.imwrite('images/tempHSV.jpg', image_three)
cv.imwrite('images/tempHLS.jpg', image_four)
cv.imwrite('images/tempLab.jpg', image_five)
cv.imwrite('images/tempLuv.jpg', image_six)
cv.imwrite('images/tempCannyXYZ.jpg', image_seven)
cv.imwrite('images/tempCannygray.jpg', image_eight)
cv.imwrite('images/XYZgray.jpg', cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2XYZ), cv.COLOR_BGR2GRAY))
cv.imwrite('images/gray.jpg', cv.cvtColor(image, cv.COLOR_BGR2GRAY))
cv.imwrite('images/tempCannyHLS.jpg', image_eleven)

image_twelve = cv.cornerHarris(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 2, 5, 0.04)
cv.imwrite('images/tempHarris.jpg', image_twelve)
