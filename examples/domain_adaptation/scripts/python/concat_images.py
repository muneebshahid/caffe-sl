import cv2
import numpy as np
import glob 
import random
import os

#Resize image.
image_size = (256, 256)

#number of instances.
N = 5

proj_home = os.getenv('PROJ_HOME')
source_files_path = proj_home + '/data/train/original_images/'
processed_files_path = proj_home + '/data/train/processed_images/'

image_files = glob.glob(source_files_path + '*.jpg')
files_num = np.size(image_files)

for image_file in range(N):
	i = random.randrange(files_num)
	j = random.randrange(files_num)
	
	img1 = cv2.resize(cv2.imread(image_files[i]), image_size)
	img2 = cv2.resize(cv2.imread(image_files[j]), image_size)
	img3 = np.vstack((img1, img2))

	cv2.imwrite(processed_files_path + '/' + str(i) + str(j) +'.jpg' , img3)

execfile('create_txt.py')