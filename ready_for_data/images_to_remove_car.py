import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

paths = glob.glob('./original_images/*.jpg')

for path in paths:
	img = cv2.imread(path)	
	print path
	img[70:290, 55:164] = 0

	cv2.imwrite('./images/' + path.split('/')[2], img)

print "Cars in original images are removed !!"