import numpy as np
import caffe


class FeatureExtractor:
    __net = None
    __transformer = None
    __transformer_key = None
    __input_layer = None
    __batch_size = None

    def __init__(self, model_path, deploy_path, mean_binary_path, input_layer, gpu_mode=True, gpu_id=0):
        if gpu_mode:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
        self.__transformer_key = 'data_'
        self.__input_layer = input_layer
        # convert binary proto to numpy array
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.ParseFromString(open(mean_binary_path, 'rb').read())
        mean_data = np.array(caffe.io.blobproto_to_array(blob))

        # create net instance.
        self.__net = caffe.Net(deploy_path, model_path, caffe.TEST)
        # create data transformer.
        self.__transformer = caffe.io.Transformer({self.__transformer_key: self.__net.blobs[self.__input_layer].data
                                                  .shape})
        # swap channels with width and height
        self.__transformer.set_transpose(self.__transformer_key, (2, 0, 1))
        self.__transformer.set_mean(self.__transformer_key, mean_data[0].mean(1).mean(1))
        # set max scale i.e  0 - 255
        self.__transformer.set_raw_scale(self.__transformer_key, 255)
        # switch to bgr from rgb
        self.__transformer.set_channel_swap(self.__transformer_key, (2, 1, 0))

    def set_batch_dim(self, batch_size, c=3, h=227, w=227):
            print 'prior batch size: {0}'.format(self.__net.blobs[self.__input_layer].data.shape)
            self.__net.blobs[self.__input_layer].reshape(batch_size, c, h, w)
            print 'curr batch size: {0}'.format(self.__net.blobs[self.__input_layer].data.shape)

    def extract(self, images, blob_keys):
        assert len(images) == self.__net.blobs[self.__input_layer].data.shape[0]
        for i, image in enumerate(images):
            image = self.__transformer.preprocess(self.__transformer_key, caffe.io.load_image(image))
            self.__net.blobs[self.__input_layer].data[i] = image
        print 'forwarding....'
        return self.__net.forward(blob_keys)
