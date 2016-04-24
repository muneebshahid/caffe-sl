import caffe
import numpy as np
import os_helper as osh

caffe.set_mode_gpu()
caffe_root = osh.get_env_var('CAFFE_ROOT')
data = caffe_root + '/../data/'
results = caffe_root + '/../results/'
mean_file = data + 'models/alexnet/pretrained/places205CNN_mean.binaryproto'
deploy = caffe_root + '/examples/domain_adaptation/network/deploy.prototxt'
im1 = data + 'images/orig/freiburg/summer/imageCompressedCam0_0003904.jpg'
im2 = data + 'images/orig/freiburg/winter/imageCompressedCam0_0016412.jpg'
blob = caffe.proto.caffe_pb2.BlobProto()
mean_data = open(data + 'models/alexnet/pretrained/places205CNN_mean.binaryproto', 'rb').read()
caffe_model = results + '/curr.caffemodel'
blob.ParseFromString(mean_data)
arr = np.array(caffe.io.blobproto_to_array(blob))
#net1 = caffe.Net(deploy, caffe.TEST)
net1 = caffe.Net(deploy, caffe_model, caffe.TEST)

transformer = caffe.io.Transformer({'data_': net1.blobs['data_1'].data.shape})
transformer.set_transpose('data_', (2, 0, 1))
transformer.set_mean('data_', arr[0].mean(1).mean(1))
transformer.set_raw_scale('data_', 255)
transformer.set_channel_swap('data_', (2, 1, 0))


def output(net, t, img1, img2):
    img1 = t.preprocess('data_', caffe.io.load_image(img1))
    img2 = t.preprocess('data_', caffe.io.load_image(img2))
    net.blobs['data_1'].data[...] = img1
    net.blobs['data_2'].data[...] = img2
    return net.forward()


def print_r(img1, img2):
    r1 = output(net1, transformer, img1, img2)
    #r2 = output(net2, transformer, img1, img2)
    print '-----------------------------------'
    print "trained: ", r1
    print 'distance: ', np.linalg.norm(r1['fc8_n'] - r1['fc8_n_p'])
    print '-----------------------------------'
    #print "trained: ", r2
    #print 'distance: ', np.linalg.norm(r2['fc8_n'] - r2['fc8_n_p'])

