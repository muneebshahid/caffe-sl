echo "Processing freiburg export file"
python "$PROJ_HOME/scripts/python/process_freiburg_export_file.py"

echo "Creating data txt files"
python "$PROJ_HOME/scripts/python/create_dataset_txt.py"

if [ "$1" = "all" ] || [ "$1" = "source_target" ] || [ "$2" = "source_target" ]
then
	echo "Creating source lmdbs"
	sh "$PROJ_HOME/scripts/shell/create_lmdbs_resize.sh" "source"
	echo "Creating target lmdbs"
	sh "$PROJ_HOME/scripts/shell/create_lmdbs_resize.sh" "target"
fi
if [ "$1" = "all" ] || [ "$1" = "train" ] || [ "$2" = "train" ]
then
	echo "Creating train lmdbs"
	sh $PROJ_HOME"/scripts/shell/create_lmdbs_resize.sh" "train"
fi
if [ "$1" = "all" ] || [ "$1" = "test" ] || [ "$2" = "test" ]
then
	echo "Creating test lmdbs"
	sh "$PROJ_HOME/scripts/shell/create_lmdbs_resize.sh" "test"
else
	echo "wrong or no lmdb param provided. aborting...."
fi


