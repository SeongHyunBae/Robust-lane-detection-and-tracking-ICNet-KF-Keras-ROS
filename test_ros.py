import cv2
import numpy as np
import time
import json
import copy
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

net = model.build_bn(224, 448, 12, train=True)
net.load_weights('output/weights.133-0.964.hdf5', by_name=True)

# net.save_weights('./model.h5')

image_height = net.inputs[0].shape[2]
image_width = net.inputs[0].shape[1]

bridge = CvBridge()
graph = tf.get_default_graph()

def avm_callback(msg):
	np_img = bridge.imgmsg_to_cv2(msg, "bgr8")
	img = copy.copy(np_img[60:420,:224,:])
	img[70:290, 55:164] = 0
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

	cv2.imshow('img', img)
	cv2.imshow('output', output)
	cv2.waitKey(1)

if __name__ == '__main__':
	rospy.init_node('avm_segmentation')
	rospy.Subscriber('/pub_avm_image', Image, avm_callback)
	rospy.spin()
	cv2.destroyAllWindows()