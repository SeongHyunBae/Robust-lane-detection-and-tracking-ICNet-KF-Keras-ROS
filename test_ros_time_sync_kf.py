import cv2
import numpy as np
import time
import json
import copy
import imutils
from scipy.signal import find_peaks

from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import load_model
import tensorflow as tf

from utils import apply_color_map
import model
import kalman

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters

from std_msgs.msg import Float64

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

net = model.build_bn(320, 960, 11, train=True)
net.load_weights('output/weights.232-0.970.hdf5', by_name=True)

# net.save_weights('./model.h5')

image_height = net.inputs[0].shape[2]
image_width = net.inputs[0].shape[1]

bridge = CvBridge()
graph = tf.get_default_graph()


# label info
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

label_classes = ['white', 'yellow', 'bus_line', 'stop_line', 'bike_line', 'road_marker', 'crosswalk', 'bust', 'bus_stop', 'zigzag', 'background']
select_classes = ['white', 'yellow', 'bus_line']

def select_class(img):
	select_img = np.zeros_like(img[:,:,0])

	for c in select_classes:
		color = label[c]
		cond0 = img[:,:,0] == color[2]
		cond1 = img[:,:,1] == color[1]
		cond2 = img[:,:,2] == color[0]
		cond = cond0 & cond1 & cond2
		select_img[cond] = 255

	return select_img

def find_peak(roi):
	mask = roi[:,:] != 0
	x = np.sum(mask, axis=0)
	peaks, _ = find_peaks(x, height=70, distance=10)
	
	return peaks

def lane_detection(img, prediction):
	global kf
	global pt_y
	global age

	max_age = 10

	prediction = select_class(prediction)

	h, w = prediction.shape
	left = copy.copy(prediction)
	right = copy.copy(prediction)

	left[450:,180] = 0
	right[450:,:179] = 0

	left_peaks = find_peak(left[:,:178])
	right_peaks = find_peak(right[:,180:])
	right_peaks += 180
	standard = 78

	fit_prev_left=[]
	fit_prev_right=[]
	fit_sum_left=0
	fit_sum_right=0
	prevFrameCount=6

	nwindows = 18
	window_height = 55

	pt_x = []

	predict = dict()

	if len(left_peaks) > 0 and len(right_peaks) > 0:
		age = 0
		
		left_index = np.argmin(abs(left_peaks - standard))
		left_base_x = left_peaks[left_index]
		left_base_y = 900
		cv2.circle(img, (left_base_x, left_base_y), 10, (255,255,0), -1)

		right_index = np.argmin(abs(right_peaks - (359 - standard)))
		right_base_x = right_peaks[right_index]
		right_base_y = 900
		cv2.circle(img, (right_base_x, right_base_y), 10, (255,0,255), -1)

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

		for i in range(nwindows/3):
			left_mid_x = int(np.mean(left_pt[i*3:i*3+3]))
			left_mid_y = h_list[i*3 + 1]
			# cv2.circle(img, (left_mid_x, left_mid_y), 10, (255,0,0), -1)

			right_mid_x = int(np.mean(right_pt[i*3:i*3+3]))
			right_mid_y = h_list[i*3 + 1]
			# cv2.circle(img, (right_mid_x, right_mid_y), 10, (255,0,0), -1)

			# cv2.circle(img, ((left_mid_x + right_mid_x)/2, (left_mid_y + right_mid_y)/2), 10, (0,0,255), -1)
			if len(pt_y) < nwindows/3:
				pt_y.append((left_mid_y + right_mid_y)/2)

			predict[i] = np.dot(H,  kf[i].predict())[0]
			# cv2.circle(img, (predict, (left_mid_y + right_mid_y)/2), 10, (0,255,0), -1)
			z = (left_mid_x + right_mid_x)/2
			kf[i].update(z)
		
	else:
		age += 1
		for i in range(nwindows/3):
			predict[i] = np.dot(H,  kf[i].predict())[0]
			kf[i].update(predict[i])


	if age < max_age:
		f2 = lambda y, a, b, c: a*y**2 + b*y + c

		for i in range(nwindows/3):
			pt_x.append(predict[i])

		try:
			a, b, c = np.polyfit(pt_y, pt_x, 2)
			pre_x = None
			pre_y = None

			for i in range(21):
				y = 0 + i * 50
				x = int(f2(y, a, b, c))

				if i == 0:
					if x - 180 > 0:
						angle = np.arctan((x-180)/400.)*180/3.14
						print 'angle : {}'.format(angle)
						angle_pub.publish(angle)
					else:
						angle = -np.arctan((180-x)/400.)*180/3.14
						print 'angle : {}'.format(angle)
						angle_pub.publish(angle)

				if i != 0:
					cv2.line(img, (pre_x, pre_y), (x, y), (255, 0, 0), 5)

				pre_x = x
				pre_y = y

		except:
			print 'No lines!'
	return img

def stop_line(img, prediction):
	h, w, c = img.shape

	cond1 = prediction[:,:,0] == label['stop_line'][2]
	cond2 = prediction[:,:,1] == label['stop_line'][1]
	cond3 = prediction[:,:,2] == label['stop_line'][0]
	cond = cond1 & cond2 & cond3

	row, col = np.where(cond)
	
	if len(row) > 3000:
		mid = int(np.average(row))
		cv2.rectangle(img, (0, mid-30), (w, mid+30), (0,0,255), 3)
		cv2.putText(img, 'STOP', (w/2-30, mid+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)

	return img

def undistort(img):
	DIM = (640, 403)
	K = np.array([[322.8344353809592, 0.0, 322.208290306497], [0.0, 322.75932679455855, 213.76296628754383], [0.0, 0.0, 1.0]])
	D = np.array([[-0.05623399470296618], [0.02450383081374598], [-0.02068599777427462], [0.007061197363889161]])

	h,w = img.shape[:2]
	map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
	undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	
	return undistorted_img

def top_view(img):
	IMAGE_H = 450
	IMAGE_W = 359

	src = np.float32([[220, 242], [36, 328], [420, 242], [604, 328]])
	dst = np.float32([[0, 0], [0, IMAGE_H], [IMAGE_W, 0], [IMAGE_W, IMAGE_H]])
	M = cv2.getPerspectiveTransform(src, dst) 

	warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) 

	warped_img = warped_img[:394,:,:]
	
	return warped_img

def callback(avm, front):
	# avm shape : (640, 359, 3)
	# front shape : (394, 359, 3)
	# avm+front shape : (1034, 359, 3)

	np_avm = bridge.imgmsg_to_cv2(avm, "bgr8")

	np_avm = np_avm[60:419,:,:]
	np_avm = np.rot90(np_avm, 3)

	np_avm[138:515, 124:240,:] = 0
	np_avm[246:270, 110:124,:] = 0
	np_avm[246:270, 240:254,:] = 0

	np_front = bridge.imgmsg_to_cv2(front, "bgr8")
	np_front = cv2.resize(np_front, (640,403))

	np_front = undistort(np_front)
	np_front = top_view(np_front)
	
	img = np.concatenate((np_front, np_avm), axis=0)
	h, w, c = img.shape

	img_ = copy.copy(cv2.resize(img, (320, 960)))


	x = np.array([cv2.resize(img_, (image_height, image_width))])

	start_time = time.time()

	with graph.as_default():
		y = net.predict(np.array(x), batch_size=1)

	with open('datasets/mapillary/config.json') as config_file:
		config = json.load(config_file)
	labels = config['labels']

	output = apply_color_map(np.argmax(y[0], axis=-1), labels)

	output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
	output = cv2.resize(output, (w, h))

	stop_result = stop_line(copy.copy(img), copy.copy(output))
	lane_result = lane_detection(copy.copy(stop_result), copy.copy(output))

	duration = time.time() - start_time
	print '{} FPS'.format(1.0/duration)

	cv2.imshow('img', img)
	cv2.imshow('output', output)
	# cv2.imshow('stop_result', stop_result)
	cv2.imshow('lane_result', lane_result)
	cv2.waitKey(1)


if __name__ == '__main__':
	pt_y = []
	age = 0

	dt = 1.0/30
	A = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
	H = np.array([1, 0, 0]).reshape(1, 3)
	Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
	R = np.array([0.5]).reshape(1, 1)

	kf = dict()
	for i in range(6):
		kf[i] = kalman.KalmanFilter(A = A, H = H, Q = Q, R = R)


	rospy.init_node('avm_segmentation')
	avm_sub = message_filters.Subscriber('/pub_avm_image', Image)
	front_sub = message_filters.Subscriber('/gmsl_camera/port_0/cam_0/image_raw', Image)
	angle_pub = rospy.Publisher('/lane_angle', Float64, queue_size=1)
	ts = message_filters.ApproximateTimeSynchronizer([avm_sub, front_sub], 10, 0.1, allow_headerless=True)
	ts.registerCallback(callback)
	rospy.spin()
	cv2.destroyAllWindows()







