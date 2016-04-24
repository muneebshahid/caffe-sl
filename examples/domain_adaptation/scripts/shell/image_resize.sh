original_dir=$1
resized_dir=$2
for image in $original_dir/*.jpg; do
	echo "processed $image"
	convert $image -resize 256x256\! $resized_dir/$(basename $image)
done	
