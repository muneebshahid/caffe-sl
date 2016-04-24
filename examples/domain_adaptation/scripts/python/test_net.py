import caffe
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os_helper as osh

caffe.set_mode_gpu()

path1 = '../data/images/michigan/uncompressed/aug/0000010126.tiff'
path2 = '../data/images/michigan/uncompressed/aug/000002406.tiff'

caffe_root = osh.get_env_var('CAFFE_ROOT')
net = caffe.Net(caffe_root + '/examples/domain_adaptation/\
        network/test_net/test.prototxt', caffe.TEST)

im1 = np.array(Image.open(path1)).transpose(2, 0, 1)
im2 = np.array(Image.open(path2)).transpose(2, 0, 1)

im_input1 = im1[np.newaxis, np.newaxis, :, :]
im_input2 = im2[np.newaxis, np.newaxis, :, :]

net.blobs['data_1'].reshape(*im_input1.shape)
net.blobs['data_2'].reshape(*im_input2.shape)

net.blobs['data_1'].data[...] = im_input1
net.blobs['data_2'].data[...] = im_input2

out = net.forward()
