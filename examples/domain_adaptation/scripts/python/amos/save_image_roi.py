from glob import glob
import ntpath
import cv2
import sys
import os
import shutil
import random
import argparse

skip_dark = False
parser = argparse.ArgumentParser()
parser.add_argument("camera_number", help="AMOS camera number")
parser.add_argument("camera_month", help="AMOS camera month")
parser.add_argument("source_folder_path", help="Path to the source folder")
parser.add_argument("target_folder_path", help="Path to the target folder")
args = parser.parse_args()

valid_camera_nums = ['190', '344', '420', '461', '441', '546','554', \
					'913', '1025', '1164', '1626', '2016', \
					'2028', '2208', '4829', '5706', '7233', \
					'7271', '10888', '19735', '21656']

camera_number = args.camera_number
camera_month = args.camera_month

if args.camera_number not in valid_camera_nums:
	print 'Invalid Camera Number'
	sys.exit()

sub_dir = '/' + camera_number + '/' + camera_month + '/'
source_folder_path = args.source_folder_path + sub_dir
target_folder_path = args.target_folder_path + sub_dir

if not os.path.exists(source_folder_path):
	print 'Invalid Folder Path'
	sys.exit()

if os.path.exists(target_folder_path):
	shutil.rmtree(target_folder_path)

os.makedirs(target_folder_path)

files = glob( source_folder_path + '/*.jpg')
total_files = str(len(files))
print 'Processing....\n'

height = {'start': 0, 'end': 0}
width = {'start': 0, 'end': 0}
if camera_number == '554' or camera_number == '1025':			
    	height['start'] = 12
elif camera_number == '913':
	height['start'] = 8
	width['end'] = -2
elif camera_number == '1164' or camera_number == '21656':
	height['start'] = 16
elif camera_number == '1626':
	height['end'] = -14
elif camera_number == '2016' or camera_number == '2028':
	height['start'] = 35
	height['end'] = -20
elif camera_number == '2208':
	height['start'] = 34
	height['end'] = -60
	width['start'] = 5
	width['end'] = -5
elif camera_number == '4829':
	height['end'] = -12
elif camera_number == '5706':
	height['start'] = 15    
elif camera_number == '7233':
	height['start'] = 65
elif camera_number == '7271':
	height['start'] = 19
elif camera_number == '10888':
	height['start'] = 8
elif camera_number == '19735':
	height['start'] = 40
elif camera_number == '441':
	skip_dark = True
elif camera_number == '190' \
	or camera_number == '344' \
	or camera_number == '420' \
	or camera_number == '461' \
	or camera_number == '546':
	height['end'] = -15
else:
	print 'Corresponding camera not found'

im = cv2.imread(files[0])
if height['end'] == 0:
	height['end'] = im.shape[0]
if width['end'] == 0:
	width['end'] = im.shape[1]

for i, im_file in enumerate(files):
	file_name = ntpath.basename(im_file)
	path = target_folder_path + file_name
	im = cv2.imread(im_file)	
	if skip_dark and im.mean() < 50:
		continue	
	if random.randrange(0, 11) == 0:
		shutil.copy(im_file, target_folder_path + 'o-' + file_name)		
	cv2.imwrite(path, im[height['start']:height['end'],\
    					 width['start']:width['end'], :])    
	if i % 100 == 0:
		print str(i) + '/' + total_files