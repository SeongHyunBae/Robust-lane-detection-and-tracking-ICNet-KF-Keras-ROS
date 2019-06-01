import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

label = {
			'white':(255, 255, 255), # 0
			'yellow':(255, 255, 0), # 1
			'bus_line':(0, 0, 255), # 2
			'stop_line':(255, 0, 0), # 3
			'bike_line':(255, 155, 0), # 4
			'road_marker':(255, 155, 155), # 5 
			'crosswalk':(0, 85, 0), # 6
			'bust':(200, 255, 200), # 7
			'bus_stop':(255, 200, 255), # 8
			'zigzag':(155, 155, 255), # 9
			'background':(0, 0, 0) # 10
		 }

label_class = ['white', 'yellow', 'bus_line', 'stop_line', 'bike_line', 'road_marker', 'crosswalk', 'bust', 'bus_stop', 'zigzag', 'background']

paths = glob.glob('./labels/*.png')

for path in paths:
	img = cv2.imread(path)

	h, w, c = img.shape

	img2 = np.zeros_like(img)

	for i, key in enumerate(label_class):
		R, G, B = label[key]

		cond1 = img[:,:,2] == R
		cond2 = img[:,:,1] == G
		cond3 = img[:,:,0] == B

		cond = cond1 & cond2 & cond3

		row, col = np.where(cond)

		try:
			img2[row,col,0] = i
			img2[row,col,1] = i
			img2[row,col,2] = i
		except:
			pass

	cv2.imwrite('./instances/' + path.split('/')[2], img2)

print "Labels are changed to instances !!"

