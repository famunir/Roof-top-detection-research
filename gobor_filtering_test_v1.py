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

clahe = cv2.createCLAHE(clipLimit = 40.0, tileGridSize = (10, 10))
AHistEqual_image = clahe.apply(combined_image)
cv2.imwrite('images/CLAHE.jpg', AHistEqual_image)
smooth_combined = cv2.GaussianBlur(AHistEqual_image, (9,9), 2)
im_canny_combined = cv2.Canny(smooth_combined,25,30)
cv2.imwrite('images/canny.jpg', im_canny_combined)

kernel = np.ones((7,7),np.uint8)
#im_closed_canny = cv2.morphologyEx(im_canny_combined, cv2.MORPH_CLOSE, kernel)
im_dilate_canny = cv2.dilate(im_canny_combined, kernel, iterations = 1)
cv2.imwrite('images/dilate.jpg', im_dilate_canny)
kernel = np.ones((5,5),np.uint8)
im_erode_closed = cv2.erode(im_dilate_canny, kernel, iterations = 1)
cv2.imwrite('images/erode_after_dialate.jpg', im_erode_closed)

approx_conts = [];
ret, threshold = cv2.threshold(im_erode_closed, 10, 12, cv2.THRESH_OTSU, None)
img2, contuors, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
for cnt in contuors:
    approx_conts.append(cv2.approxPolyDP(cnt, 0.00005 * cv2.arcLength(cnt,True), True))
im_with_conts = cv2.drawContours(original_image, approx_conts, -1, (0,255,0), 3)
cv2.imwrite('images/imageWithConts.jpg', im_with_conts)


