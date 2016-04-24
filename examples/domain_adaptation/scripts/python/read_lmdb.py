import numpy as np
import lmdb
import caffe
import matplotlib.image as mpimg

lmdb_path = '../../data/lmdb/lmdb_train'
lmdb_env = lmdb.open(lmdb_path, readonly=True)
# lmdb_txn = lmdb_env.begin()
# lmdb_cursor = lmdb_txn.cursor()
# lmdb_cursor.get('{:0>10d}'.format(1))
# value = lmdb_cursor.value()
# key = lmdb_cursor.key()

# datum = caffe.proto.caffe_pb2.Datum()
# datum.ParseFromString(value)
# #image = np.zeros((datum.channels, datum.height, datum.width))
# image = caffe.io.datum_to_array(datum)
# image = np.transpose(image, (1, 2, 0))
# iamge = image[:, :, (2, 1, 0)]
# image = image.astype(np.uint8)

# mpimg.imsave('out.jpg', image)

with lmdb_env.begin() as lmdb_txn:
    raw_datum = lmdb_txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.fromstring(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
y = datum.label

print y

# with lmdb_env.begin() as lmdb_txn:
#     cursor = lmdb_txn.cursor()
#     for key, value in cursor:
#         print(key, value)