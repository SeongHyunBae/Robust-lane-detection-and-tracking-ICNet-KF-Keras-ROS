import cv2
import numpy as np
import glob
import shutil

paths = glob.glob('./images/*.jpg')

l = len(paths)
val_len = int(l * 0.1)

ran_list = []

for i in range(l):
	ran = np.random.randint(l, size=1)
	ran = ran[0]

	if ran not in ran_list:
		ran_list.append(ran)

	if len(ran_list) == val_len:
		break
		
for i in ran_list:
	shutil.move(paths[i], './images3/' + paths[i].split('/')[2])
	shutil.move('./instances/{}.png'.format(paths[i].split('/')[2].split('.')[0]), './instances3/{}.png'.format(paths[i].split('/')[2].split('.')[0]))