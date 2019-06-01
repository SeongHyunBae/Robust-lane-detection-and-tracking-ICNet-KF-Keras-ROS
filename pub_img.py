import cv2
import numpy as np
import rospy
import roslib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

cap = cv2.VideoCapture(0)

pub_avm_image = rospy.Publisher('/pub_avm_image', Image, queue_size=1)

bridge = CvBridge()

rospy.init_node('avm_image', anonymous=True)

while(1):
	ret, frame = cap.read()

	cv2.imshow('frame', frame)
	
	msg = bridge.cv2_to_imgmsg(frame, 'bgr8')
	
	time = rospy.Time.now()
	#time = rospy.get_rostime()
	#print 'secs : {}     nsecs : {}'.format(time.secs, time.nsecs)
	msg.header.stamp = time

	pub_avm_image.publish(msg)

	key = cv2.waitKey(1) & 0xff

	if key == ord('q'):
		break
