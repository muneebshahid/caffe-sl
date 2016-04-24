model=""
snapshot=""
network=$CAFFE_ROOT"/examples/domain_adaptation/network/"
log_dir=$CAFFE_ROOT"/data/domain_adaptation_data/logs"
if [ "$1" = "finetune" ]
then
	solver=$network"pretrained_solver.prototxt"
	if [ "$2" = "default" ]
	then
		if [ "$3" = "alex" ]
		then
			model=$CAFFE_ROOT"/../data/models/alexnet/pretrained/places205CNN_iter_300000_upgraded.caffemodel"
		elif [ "$3" = "vgg" ]
		then
			model=$CAFFE_ROOT"/../data/models/vgg/siat_scene_vgg_11.caffemodel"
		else
			echo "wrong 3rd param"
			exit
		fi	
	elif [ "$2" = "trained" ]
	then
		if [ "$3" = "w" ]
		then
			model="$4"
		elif [ "$3" = "s" ]
		then
			snapshot="$4"
		else
			echo "wrong 3rd param"
			exit
		fi
	else
		echo "wrong 2nd param"
		exit
	fi
	$CAFFE_ROOT/build/tools/caffe train --solver=$solver --weights=$model --snapshot=$snapshot --log_dir=$log_dir -gpu 0
elif [ "$1" = "scratch" ]
then	
	solver=$network"scratch_solver.prototxt"
	$CAFFE_ROOT/build/tools/caffe train --solver=$solver --log_dir=$log_dir -gpu 0
else
	echo "wrong first param"
fi
