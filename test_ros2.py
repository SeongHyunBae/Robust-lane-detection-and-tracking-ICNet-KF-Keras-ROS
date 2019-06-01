import cv2
import numpy as np
import time
import json
import copy
import imutils

from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import load_model
import tensorflow as tf

from utils import apply_color_map
import model

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

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

label_class = ['white', 'yellow', 'bus_line', 'stop_line', 'bike_line', 'road_marker', 'crosswalk', 'bust', 'bus_stop', 'zigzag', 'background']


# def white_yello(img):
# 	img2 = np.zeros_like(img[:,:,0])

# 	for c in label_class:
# 		color = label[c]
# 		cond0 = img[:,:,0] == color[2]
# 		cond1 = img[:,:,1] == color[1]
# 		cond2 = img[:,:,2] == color[0]
# 		cond = cond0 & cond1 & cond2
# 		img2[cond] = 255

# 	return img2

def stop_line(img):
	h, w, c = img.shape

	cond1 = img[:,:,0] == label['stop_line'][2]
	cond2 = img[:,:,1] == label['stop_line'][1]
	cond3 = img[:,:,2] == label['stop_line'][0]
	cond = cond1 & cond2 & cond3

	row, col = np.where(cond)
	
	if len(row) > 2000:
		mid = int(np.average(row))
		cv2.rectangle(img, (0, mid-30), (w, mid+30), (0,255,0), 3)
		cv2.putText(img, 'STOP', (w/2-30, mid+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)
	# cv2.imshow('stop', img)

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

def avm_callback(msg):
	global avm

	np_img = bridge.imgmsg_to_cv2(msg, "bgr8")

	np_img = np_img[60:419,:,:]
	np_img = np.rot90(np_img, 3)

	np_img[138:515, 124:240,:] = 0
	np_img[246:270, 110:124,:] = 0
	np_img[246:270, 240:254,:] = 0
	
	avm = np_img
		
def front_callback(msg):
	global avm
	global count

	if count % 2 == 0:
		np_img = bridge.imgmsg_to_cv2(msg, "bgr8")
		front = cv2.resize(np_img, (640,403))

		front = undistort(front)
		front = top_view(front)

		img = np.concatenate((front, avm), axis=0)

		img = copy.copy(cv2.resize(img, (320, 960)))

		h, w, c = img.shape

		x = np.array([cv2.resize(img, (image_height, image_width))])

		start_time = time.time()

		with graph.as_default():
			y = net.predict(np.array(x), batch_size=1)

		duration = time.time() - start_time

		print '{} FPS'.format(1.0/duration)

		# Save output image
		with open('datasets/mapillary/config.json') as config_file:
			config = json.load(config_file)
		labels = config['labels']

		output = apply_color_map(np.argmax(y[0], axis=-1), labels)

		output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
		output = cv2.resize(output, (w, h))
		stop_line(output[:490,70:w-70,:])
		cv2.imshow('output', output)
		cv2.imshow('img', img)
		cv2.waitKey(1)
	count += 1





if __name__ == '__main__':
	count = 0
	avm = np.zeros((640, 359, 3))
	rospy.init_node('avm_segmentation')
	rospy.Subscriber('/pub_avm_image', Image, avm_callback)
	rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw', Image, front_callback)
	rospy.spin()
	cv2.destroyAllWindows()







