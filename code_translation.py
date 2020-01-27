#!/usr/bin/env python

import cv2
import sys
import numpy as np
from PIL import Image

image_path = sys.argv[1]
original_image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

var = 0.1
thresh = 0.45

hsv_one = hsv_image[:,:,0]
hsv_one.astype(float)
hsv_one = hsv_one > thresh
hsv_two = hsv_image[:,:,1]
hsv_two.astype(float)
hsv_two = hsv_two > thresh
hsv_three = hsv_image[:,:,2]
hsv_three.astype(float)
hsv_three = hsv_three > thresh
#cv2.imwrite('hsv_one.jpg',hsv_one)

merge_one = np.dstack((hsv_one, hsv_two))
merge_two = np.dstack((merge_one, hsv_three))

processed_image = cv2.cvtColor(merge_two, cv2.COLOR_BGR2GRAY)



