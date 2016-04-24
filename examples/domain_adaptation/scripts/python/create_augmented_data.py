import os_helper as osh
import sys
import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt


def show(ims, titles):
    for i, im in enumerate(ims):
        plt.subplot(220 + i),plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), plt.title(titles[i])
    plt.show()
    return


def translate(im):
    pos = np.random.random_integers(10, 30, 1)
    neg = np.random.random_integers(-30, -10, 1)
    im1 = np.float32([[1, 0, pos],[0, 1, pos]])
    im2 = np.float32([[1, 0, pos],[0, 1, neg]])
    im3 = np.float32([[1, 0, neg],[0, 1, pos]])
    im4 = np.float32([[1, 0, neg],[0, 1, neg]])
    M = [im1, im2, im3, im4]
    dst = [cv2.warpAffine(im, m, (cols, rows)) for m in M]
    return dst


def rotate(im):
    rot_deg = np.random.random_integers(0, 360, aug_num)
    M = [cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1) for deg in rot_deg]
    dst = [cv2.warpAffine(im, m, (cols,rows)) for m in M]
    return dst


def affine_transform(im):
    point_1 = [np.random.random_integers(30, 70, 3).tolist() for i in range(0, aug_num)]
    point_2 = [np.random.random_integers(180, 220, 3).tolist() for i in range(0, aug_num)]
    pts1 = np.float32([[50, 50], [200, 50], [200, 200]])
    pts2 = [np.float32([[pt1[0], pt1[1]], [pt2[0], pt1[2]], [pt2[1], pt2[2]]])
            for pt1, pt2 in zip(point_1, point_2)]
    M = [cv2.getAffineTransform(pts1, pt2) for pt2 in pts2]
    dst = [cv2.warpAffine(im, m,(cols, rows)) for m in M]
    return dst


def perspective_transform(im):
    point_1 = [np.random.random_integers(35, 75, 4).tolist() for i in range(0, aug_num)]
    point_2 = [np.random.random_integers(200, 240, 4).tolist() for i in range(0, aug_num)]
    pts1 = np.float32([[55, 55], [220, 55], [55, 220], [220, 220]])
    pts2 = [np.float32([[pt1[0], pt1[1]], [pt2[0], pt1[2]], [pt1[3], pt2[1]], [pt2[2], pt2[3]]])
            for pt1, pt2 in zip(point_1, point_2)]
    M = [cv2.getPerspectiveTransform(pts1, pt) for pt in pts2]
    dst = [cv2.warpPerspective(im, m, (cols, rows)) for m in M]
    return dst


def process(im):
    return {'tra': translate(copy.deepcopy(im)), 'rot': rotate(copy.deepcopy(im)),
            'aff': affine_transform(copy.deepcopy(im)), 'per': perspective_transform(copy.deepcopy(im))}


def write(path, orig_name, augmented_ims):
    name, ext = orig_name.split('.')
    for key in augmented_ims:
        for i, ims in enumerate(augmented_ims[key]):
            cv2.imwrite(path + name + '-' + key + str(i) + '.' + ext, ims)


def create_augmented_data(folders, root_folder_path):
    root_orig_path = root_folder_path + 'orig/'
    root_augmented_path = root_folder_path + 'augmented/'
    if not osh.path_exists(root_augmented_path):
        osh.make_dir(root_augmented_path)

    total_file_num = 0
    for key in folders:
        for folder in folders[key]:
            orig_path = root_orig_path + key + '/' + folder + '/'
            total_file_num += len(osh.list_dir(orig_path))
    print 'total files found: ', total_file_num

    over_all_done = 0
    for key in folders:
        if key == 'freiburg':
            print 'processing freiburg....'
        elif key == 'michigan':
            print 'processing michigan....'
        elif key == 'fukui':
            print 'processing fukui....'
        elif key == 'nordland':
            print 'processing nordland....'

        for folder in folders[key]:
            print 'processing ' + key + ' ' + folder
            orig_path = root_orig_path + key + '/' + folder + '/'
            augm_path = root_augmented_path + key + '/' + folder + '/'
            if osh.path_exists(augm_path):
                print 'removing existing folder ', augm_path
                osh.rm_dir(augm_path)
            print 'creating folder ', augm_path
            osh.make_dir(augm_path)
            files = osh.list_dir(orig_path)
            len_files = len(files)
            for i, im_file in enumerate(files):
                if key == 'michigan' and int(im_file[:-5]) in mich_ignore:
                    print "ignoring {0}".format(im_file[:-5])
                else:
                    augmented_ims = process(cv2.imread(orig_path + im_file))
                    write(augm_path, im_file, augmented_ims)
                if i % 100 == 0:
                    print str(i) + '/' + str(len_files)
                if over_all_done % 100 == 0:
                    print 'Total progress: ' + str(over_all_done) + '/' + str(total_file_num)
                over_all_done += 1


def main():
    root_folder_path = osh.get_env_var('CAFFE_ROOT') + '/../data/images/'
    root_folder_path = root_folder_path.replace('\\', '/')
    if not osh.is_dir(root_folder_path):
        print "source folder does'nt exist, existing....."
        sys.exit()
    keys = ['freiburg', 'michigan', 'fukui']
    fukui_parent_folders = ['AU', 'SU', 'SP', 'WI']
    fukui_d_q_folders = ['db', 'query']
    fukui_child_folders = [str(i) for i in range(1, 13)]
    fukui_folders = [p_folder + '/' + dq + '/' + c_folder + '/'
                     for p_folder in fukui_parent_folders
                     for dq in fukui_d_q_folders
                     for c_folder in fukui_child_folders]
    folders = {#'freiburg': ['summer', 'winter'],
               #'michigan': ['aug', 'jan'],
               #'fukui': fukui_folders,
               'nordland': ['summer', 'winter', 'spring', 'fall']}
    print "Augmented Data Saftey Lock"
    create_augmented_data(folders, root_folder_path)


if __name__ == "__main__":
    rows, cols, channels = 256, 256, 3
    aug_num = 4
    mich_ignore = range(1264, 1272)
    mich_ignore.extend(range(1473, 1524))
    mich_ignore.extend(range(1553, 1565))
    mich_ignore.extend(range(1623, 1628))
    # indoor
    mich_ignore.extend(range(4795, 5521))
    mich_ignore.extend(range(8324, 9288))
    mich_ignore.extend(range(10095, 10677))
    mich_ignore.extend(range(11270, 11776))
    mich_ignore.extend(range(11985, 12575))
    main()
