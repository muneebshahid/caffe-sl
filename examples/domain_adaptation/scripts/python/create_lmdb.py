import numpy as np
import glob 
from PIL import Image
import cv2
import random
import lmdb
import caffe

#Number of training examples.
N = 10 

#Resize image.
image_size = (256, 256)

channels_num = 3

#Path to folder containing the training images.
train_path = '../../data/train/processed_images/'
lmdb_path = '../../data/lmdb/lmdb_train' 

train_files = glob.glob(train_path + '*.jpg')
sorted_files = sorted(train_files)
files_num = np.size(sorted_files)

X = np.zeros((N, 3, image_size[0], image_size[1]), dtype=np.uint8)
y = np.ones(N, dtype=np.int64)

map_size = X.nbytes * 10

env = lmdb.open(lmdb_path, map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for im_file in range(N):
    	
    	i = random.randrange(files_num)
    	j = random.randrange(files_num)

    	img1 = cv2.imread(sorted_files[i]).transpose(2, 0, 1)
    	img2 = cv2.imread(sorted_files[i]).transpose(2, 0, 1)
    	img3 = np.concatenate((img1,img2))
        label = {1, 2}; 

        datum = caffe.io.array_to_datum(img3, label)

        #datum = caffe.proto.caffe_pb2.Datum()
        #datum.channels = X.shape[1]
        #datum.height = X.shape[2]
        #datum.width = X.shape[3]
        #datum.data = X[im_file].tobytes()  # or .tostring() if numpy < 1.9
        #datum.label = int(y[im_file])

        txn.put('{:08}'.format(im_file), datum.SerializeToString())