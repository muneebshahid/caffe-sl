
import numpy as np
import deepdish as dd
import lmdb
import sys
import caffe
import glob 
import random
from PIL import Image
import plyvel
import os


proj_home = os.getenv('PROJ_HOME');

#Number of training examples.
N = 5 

#Resize image.
image_size = (256, 256)

channels_num = 3

#Path to folder containing the training images.
train_path = proj_home + '/data/train/processed_images/'
level_db_path = proj_home + '/data/leveldb/train/' 

train_files = glob.glob(train_path + '*.jpg')
sorted_files = sorted(train_files)
files_num = np.size(sorted_files)


# Let's pretend this is interesting data
X = np.zeros((N, 3, 160, 60), dtype=np.uint8)

# We need to prepare the database for the size. If you don't have 
# deepdish installed, just set this to something comfortably big 
# (there is little drawback to settings this comfortably big).
map_size = dd.bytesize(X) * 2.5
db = plyvel.DB(level_db_path,create_if_missing = True, error_if_exists=True, write_buffer_size=map_size)
wb=db.write_batch()

count = 0
for k in range(N):

  i = random.randrange(files_num)
  j = random.randrange(files_num)
  label_i = 1      #label from image filename
  label_j = 1
        
  if label_i == label_j:
      label = 1
  else:
      label = 0

  img1 = np.array(Image.open(sorted_files[i]))
  img1 = img1[:, :, (2, 1, 0)]
  img1 = img1.transpose((2, 0, 1))
  img2 = np.array(Image.open(sorted_files[j]))
  img2 = img2[:, :, (2, 1, 0)]
  img2 = img2.transpose((2, 0, 1))
  img3 = np.concatenate((img1,img2))

  datum = caffe.io.array_to_datum(img3, label)  

  wb.put('%08d' % count, datum.SerializeToString())
  count += 1
  if count % 64 == 0:  #64 is batch size
  # Write batch of images to database
    wb.write()
    del wb
    wb = db.write_batch()
    print 'Processed %i images.' % count
  if count % 64 != 0:
  # Write last batch of images
    wb.write()
