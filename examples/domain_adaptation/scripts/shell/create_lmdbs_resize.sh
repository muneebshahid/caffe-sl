convert_imageset="$CAFFE_ROOT/build/tools/convert_imageset"
data="$CAFFE_ROOT/data/"
lmdb="$data/domain_adaptation_data/lmdb/"
images="$data/domain_adaptation_data/images/"

if [ $1 = "complete" ]
then
	complete_lmdb=$lmdb$1
	rm -r $complete_lmdb
	$convert_imageset $images $images$1".txt" $complete_lmdb
elif [ $1 = "source" ] || [ $1 = "target" ] || [ $1 = "train" ] || [ $1 = "test" ] || [ $1 = "labels" ]
then
	if [ "$2" = "all" ]
	then
		lmdb1=$lmdb$1"1"
		lmdb2=$lmdb$1"2"
		rm -r $lmdb1
		rm -r $lmdb2
		$convert_imageset $images $images$1"1.txt" $lmdb1
		$convert_imageset $images $images$1"2.txt" $lmdb2
	elif [ "$2" = "1" ] || [ "$2" = "2" ]
	then
		$convert_imageset $images $images$1$2".txt" $lmdb$1$2
	else
		echo "wrong 2nd param"
	fi
else
	echo "wrong 1st param"
fi
