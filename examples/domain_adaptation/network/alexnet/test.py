import caffe
import numpy as np
from pylab import *

niter = 200
# losses will also be stored in the log
train_loss = np.zeros(niter)
scratch_train_loss = np.zeros(niter)

caffe.set_device(0)
caffe.set_mode_gpu()

#net = caffe.Net('places205CNN_train_val.prototxt', '../../data/models/alexnet_places/places205CNN_iter_300000_upgraded.caffemodel', caffe.TRAIN)
# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver('places205CNN_solver.prototxt')
solver.net.copy_from('../../data/models/alexnet_places/places205CNN_iter_300000_upgraded.caffemodel')
