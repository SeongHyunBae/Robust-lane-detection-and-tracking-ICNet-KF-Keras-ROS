import cv2
import glob
import numpy as np
# stopline (255, 0, 0)
# bikeline (255, 155, 0)
# crosswork (0, 85, 0)
# single_white (255, 255, 255)
# single_yellow (255, 255, 0)
# background (0, 0, 0)
# busline (0, 0, 255)
# roadmarker (255, 155, 155)
# zigzag (155, 155, 255)
# inaccessible (155, 155, 155)
# dashed_white (200, 255, 200)
# double_yellow (255, 200, 255)

label = {
		 'single_white':(255, 255, 255),
		 'dashed_white':(200, 255, 200),
		 'single_yellow':(255, 255, 0),
		 'double_yellow':(255, 200, 255),
		 'stop_line':(255, 0, 0),
		 'bike_line':(255, 155, 0),
		 'crosswalk':(0, 85, 0),
		 'bus_line':(0, 0, 255),
		 'road_marker':(255, 155, 155),
		 'zigzag':(155, 155, 255),
		 'inaccessible':(155, 155, 155),
		 'background':(0, 0, 0)
		 }

paths = glob.glob('./original_labels/*.jpg')


for path in paths:
	img = cv2.imread(path)
	
	h, w, c = img.shape

	img2 = np.zeros_like(img)

	for key in label.keys():
		margin = 50
		R, G, B = label[key]

		cond1_1 = R - margin <= img[:,:,2]
		cond1_2 = img[:,:,2] <= R + margin
		cond1 = cond1_1 & cond1_2

		cond2_1 = G - margin <= img[:,:,1]
		cond2_2 = img[:,:,1] <= G + margin
		cond2 = cond2_1 & cond2_2

		cond3_1 = B - margin <= img[:,:,0]
		cond3_2 = img[:,:,0] <= B + margin
		cond3 = cond3_1 & cond3_2

		cond = cond1 & cond2 
		cond = cond & cond3

		row, col = np.where(cond)

		try:
			img2[row,col,0] = B
			img2[row,col,1] = G
			img2[row,col,2] = R
		except:
			pass

	img2[70:290, 55:164] = 0
	cv2.imshow('img', img)
	cv2.imshow('img2', img2)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
		
	cv2.imwrite('./labels/{}.png'.format(path.split('/')[-1].split('.')[0]), img2)

print "Original labels are repainted !!"
	