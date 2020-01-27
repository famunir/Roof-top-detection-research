#!/usr/bin/env python

import cv2
import sys
import numpy as np
from PIL import Image

image_path = sys.argv[1]
original_image = cv2.imread(image_path)

# XYZ_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2XYZ)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Gabor filtering
kernel_size = 7
sigma = 3
theta = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
lambd = 0.01
gamma = 2
psi = 0
combined_image = np.zeros_like(np.float64(gray_image))

for ii in range(18):
	gobor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta[ii], lambd, gamma, psi, ktype = cv2.CV_64F)
	im_gobor_filtered = cv2.filter2D(gray_image, cv2.CV_64F, gobor_kernel)
	image_index = "images/" + `ii` + ".jpg"
	cv2.imwrite(image_index, im_gobor_filtered)
	combined_image +=  im_gobor_filtered

combined_image = np.uint8((combined_image-np.min(combined_image))/(np.max(combined_image)-np.min(combined_image))*255)
cv2.imwrite('images/combined_image.jpg', combined_image)

clahe = cv2.createCLAHE(clipLimit = 100.0, tileGridSize = (8, 8))
AHistEqual_image = clahe.apply(combined_image)
cv2.imwrite('images/CLAHE.jpg', AHistEqual_image)
#smooth_combined = cv2.GaussianBlur(AHistEqual_image, (9,9), 2)
#im_canny_combined = cv2.Canny(smooth_combined,25,30)
#cv2.imwrite('images/canny.jpg', im_canny_combined)

approx_conts = [];
#ret, threshold = cv2.threshold(im_erode_closed, 100, 255, cv2.THRESH_BINARY, None)
threshold_image = cv2.adaptiveThreshold(AHistEqual_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,71,9)
cv2.imwrite('images/thresholded.jpg', threshold_image)

kernel = np.ones((7,7),np.uint8)
im_erode = cv2.erode(threshold_image, kernel, iterations = 2)
cv2.imwrite('images/erode.jpg', im_erode)

img2, contuors, hierarchy = cv2.findContours(im_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

kernel = np.ones((7,7),np.uint8)
im_dilate = cv2.dilate(im_erode, kernel, iterations = 1)
cv2.imwrite('images/dilate.jpg', im_dilate)

img2, contuors, hierarchy = cv2.findContours(im_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
for cnt in contuors:
    approx_conts.append(cv2.approxPolyDP(cnt, 0.00005 * cv2.arcLength(cnt,True), True))
    im_with_conts = cv2.drawContours(original_image, [cnt], -1, (255*np.random.random(),255*np.random.random(), 255*np.random.random()), -1)
cv2.imwrite('images/imageWithConts.jpg', im_with_conts)


