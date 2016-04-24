data=$CAFFE_ROOT"/../data/"
$CAFFE_ROOT/build/tools/compute_image_mean $CAFFE_ROOT/data/domain_adaptation_data/lmdb/complete $data/models/alexnet/scratch/mean.binaryproto
