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

def preproStageOne(gray):
    """
    @param gray: gray scale image to use for processing
    @output: unsharp masked and bilaterlly smooth image
    """
    gaussBlur = cv2.GaussianBlur(gray, (9,9), 10.0)
    unsharpImage = cv2.addWeighted(gray, 1.5, gaussBlur, -0.5, 0, gray)
    bilaterlSmooth = cv2.bilateralFilter(unsharpImage, 9, 75, 75)
    return bilaterlSmooth
    
#################################################################################

result_image = preproStageOne(gray_image)
cv2.imwrite('result.jpg',result_image)
cv2.imwrite('result2.jpg', gray_image)
