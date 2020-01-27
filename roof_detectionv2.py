#!/usr/bin/env python

import cv2
import sys
import numpy as np
from PIL import Image

image_path = sys.argv[1]
original_image = cv2.imread(image_path)
# XYZ_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2XYZ)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

#################################################################################

def extractConts(gray, epsilon = 0.00005):
    """
    @param gray: gray scale image to use for processing
    @param epsilon: parameter for approxPolyDP
    @output approx: list of all detected contuors
    """
    clahe = cv2.createCLAHE(clipLimit = 40.0, tileGridSize = (10, 10))
    AHistEqual_image = clahe.apply(gray)
    ret, threshold = cv2.threshold(AHistEqual_image, 50, 255, cv2.THRESH_OTSU, None)
    img2, contuors, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    approx = []
    for cnt in contuors:
    	approx.append(cv2.approxPolyDP(cnt, epsilon * cv2.arcLength(cnt,True), True))
    return approx
    
#################################################################################

approxPolyMat = extractConts(gray_image)
imageWithConts = cv2.drawContours(original_image, [approxPolyMat[253]], -1, (0,255,0), 3)
print len(approxPolyMat[253])
cv2.imwrite('imageWithConts.jpg', imageWithConts)

