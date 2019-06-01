import cv2
import numpy as np
import glob
import copy
import time
from scipy.signal import find_peaks

paths = glob.glob('./img/*.png')
paths.sort()


def find_peak(roi):
	mask = np.logical_and(roi[:,:,0] != 0, roi[:,:,1] != 0, roi[:,:,2] != 0)
	x = np.sum(mask, axis=0)
	peaks, _ = find_peaks(x, height=70, distance=10)
	
	return peaks

def lane_detection(img):
	h, w, c = img.shape

	left = copy.copy(img)
	right = copy.copy(img)

	left[450:,180:,:] = 0
	right[450:,:179,:] = 0

	left_peaks = find_peak(left[:,:178,:])
	right_peaks = find_peak(right[:,180:,:])
	right_peaks += 180
	standard = 78

	window_size = (60, 50)

	fit_prev_left=[]
	fit_prev_right=[]
	fit_sum_left=0
	fit_sum_right=0
	prevFrameCount=6

	if len(left_peaks) > 0 and len(right_peaks) > 0:
		left_index = np.argmin(abs(left_peaks - standard))
		left_base_x = left_peaks[left_index]
		left_base_y = 900
		cv2.circle(img, (left_base_x, left_base_y), 10, (255,255,0), -1)

		right_index = np.argmin(abs(right_peaks - (359 - standard)))
		right_base_x = right_peaks[right_index]
		right_base_y = 900
		cv2.circle(img, (right_base_x, right_base_y), 10, (255,0,255), -1)

		nwindows = 18
		window_height = 55

		left_nonzero = left.nonzero()
		left_nonzeroy = np.array(left_nonzero[0])
		left_nonzerox = np.array(left_nonzero[1])
		right_nonzero = right.nonzero()
		right_nonzeroy = np.array(right_nonzero[0])
		right_nonzerox = np.array(right_nonzero[1])

		leftx_current = left_base_x
		rightx_current = right_base_x
		margin = 35
		minpix = 50

		left_pt = []
		right_pt = []
		h_list = []

		for window in range(nwindows):
			win_y_low = img.shape[0] - (window+1)*window_height
			win_y_high = img.shape[0] - window*window_height

			left_pt.append(leftx_current)
			right_pt.append(rightx_current)
			h_list.append((win_y_low + win_y_high) / 2)

			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin

			cv2.rectangle(img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
			cv2.rectangle(img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 

			good_left_inds = ((left_nonzeroy >= win_y_low) & (left_nonzeroy < win_y_high) & (left_nonzerox >= win_xleft_low) &  (left_nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((right_nonzeroy >= win_y_low) & (right_nonzeroy < win_y_high) & (right_nonzerox >= win_xright_low) &  (right_nonzerox < win_xright_high)).nonzero()[0]

			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(left_nonzerox[good_left_inds])*0.3 + np.max(left_nonzerox[good_left_inds])*0.7)

			if len(good_right_inds) > minpix:        
				rightx_current = np.int(np.mean(right_nonzerox[good_right_inds])*0.3 + np.min(right_nonzerox[good_right_inds])*0.7)

		for i in range(6):
			left_mid_x = int(np.mean(left_pt[i*3:i*3+3]))
			left_mid_y = h_list[i*3 + 1]
			cv2.circle(img, (left_mid_x, left_mid_y), 10, (255,0,0), -1)

			right_mid_x = int(np.mean(right_pt[i*3:i*3+3]))
			right_mid_y = h_list[i*3 + 1]
			cv2.circle(img, (right_mid_x, right_mid_y), 10, (255,0,0), -1)

			cv2.circle(img, ((left_mid_x + right_mid_x)/2, (left_mid_y + right_mid_y)/2), 10, (0,0,255), -1)

			

cv2.namedWindow('img')        
cv2.moveWindow('img', 0, 0)

for path in paths:
	img = cv2.imread(path)

	start = time.time()
	lane_detection(img)
	print 'FPS : {}'.format(1 / (time.time() - start))


	cv2.imshow('img', img)

	if cv2.waitKey(1) & 0xff == ord('q'):
		break

