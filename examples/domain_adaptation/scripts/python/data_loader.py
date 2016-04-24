import os_helper as osh
import numpy as np
import pandas as pd


class DataLoader():

    def __init__(self):
        return

    @staticmethod
    def load(key, root_folder_path):
        folder_path = root_folder_path + key + '/'
        dataset = []
        if key == 'freiburg':
            dataset = DataLoader.__load_freiburg(folder_path)
        elif key == 'michigan':
            dataset = DataLoader.__load_michigan(folder_path)
        elif key == 'fukui':
            dataset = DataLoader.__load_fukui(folder_path)
        elif key == 'alderly':
            dataset = DataLoader.__load_alderly(folder_path)
        elif key == 'kitti':
            dataset = DataLoader.__load_kitti(folder_path)
        elif key == 'nordland':
            dataset = DataLoader.__load_nordland(folder_path)
        return dataset

    @staticmethod
    def __load_freiburg(folder_path):
        dataset = []
        print "processing freiburg data....."
        with open(folder_path + 'processed_season_match.txt', "r") as data_reader:
            data_set_freiburg = [line.replace('\n', '').split(' ') for line in data_reader.readlines()]

        with open(folder_path + 'pxgps/summer_track.pxgps') as gps_reader:
            frei_gps_data = [np.array(line.replace('\n', '').split(' ')[1:], np.float64)
                              for line in gps_reader.readlines()]

            for instance in data_set_freiburg:
                j = 1
                while j < len(instance):
                    file_name, _ = osh.split_file_extension(osh.extract_name_from_path(instance[0]))
                    gps_index = int(file_name[-7:])
                    dataset.append([folder_path + instance[0],
                                     folder_path + instance[j], 1, frei_gps_data[gps_index]])
                    j += 1
        return dataset

    @staticmethod
    def __load_michigan(folder_path):
        dataset = []
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
        print 'Processing michigan data.....'
        files_michigan = [folder_path + 'aug/' + im + '.jpg'
                          for im in
                          sorted([im[:-4] for im in
                                  osh.list_dir(folder_path + 'aug/')], key=int)]

        for im_file in files_michigan:
            file_n = osh.extract_name_from_path(im_file)
            if int(file_n[:-4]) in mich_ignore:
                # print "ignoring {0}".format(file_n[:-5])
                continue
            dataset.append([im_file, im_file.replace('aug', 'jan'), 1, None])
        return dataset

    @staticmethod
    def get_fukui_im_path(image_id, gt_id, root_folder, is_query=False):
        path = root_folder + ('query/' if is_query else 'db/') + gt_id + image_id
        return path


    @staticmethod
    def __load_fukui(folder_path):
        dataset = []
        # mislabeled examples
        fukui_ignore = {'SU': ['4', '04000000'],
                        'SP': ['4', '04000000', '04000001', '04000002'],
                        'AU': ['1', '01000000', '01000001', '01000002', '01000003'],
                        'WI': ['None']}
        print 'processing fukui data.....'
        extra_im_range = 3
        instance = ['AU', 'SP', 'SU', 'WI']
        for season in instance:
            print 'processing: ', season
            season_folder = folder_path + season + '/'
            ground_truth_folder = season_folder + 'gt/'
            gts = sorted([gt[:-4] for gt in osh.list_dir(ground_truth_folder)], key=int)
            print 'creating positive examples'
            for gt in gts:
                with open(ground_truth_folder + gt + '.txt', "r") as ground_truths:
                    for line in ground_truths.readlines():
                        # skip \n
                        images = line.replace('\n', '').split(' ')
                        qu_image = images[0]
                        db_image = images[1]
                        # skip mislabeled examples
                        if gt == fukui_ignore[season][0] and qu_image in fukui_ignore[season]:
                            print 'ignoring {0}, gt {1}, season {2}'.format(qu_image, gt, season)
                            continue
                        qu_image_path = DataLoader.get_fukui_im_path(qu_image + '.jpg', gt + '/', season_folder, True)
                        db_image_path = DataLoader.get_fukui_im_path(db_image + '.jpg', gt + '/', season_folder)
                        if not osh.is_file(qu_image_path):
                            print "query not found", qu_image_path
                        if not osh.is_file(db_image_path):
                            print '---------------'
                            print 'db not found'
                            print qu_image_path
                            print db_image_path
                            print '---------------'
                        if int(db_image) < int(images[2]):
                            print 'db less than limit'
                            print qu_image_path
                            print db_image_path
                            print '----------------'
                        gt_example = [qu_image_path, db_image_path, 1, None]

                        dataset.append(gt_example)
                        int_db_image = int(db_image)
                        im_range = range(int(images[2]), int(images[3]) + 1)

                        # append images before if possible
                        for im in range(int_db_image - extra_im_range, int_db_image):
                            if im in im_range:
                                db_image_path = DataLoader.get_fukui_im_path('0' + str(im) + '.jpg', gt + '/', season_folder)
                                if osh.is_file(db_image_path):
                                    gt_example = [qu_image_path, db_image_path, 1, None]
                                    dataset.append(gt_example)

                        # append images after if possible
                        for im in range(int_db_image + 1, int_db_image + extra_im_range + 1):
                            if im in im_range:
                                db_image_path = DataLoader.get_fukui_im_path('0' + str(im) + '.jpg', gt + '/', season_folder)
                                if osh.is_file(db_image_path):
                                    gt_example = [qu_image_path, db_image_path, 1, None]
                                    dataset.append(gt_example)
        return dataset

    @staticmethod
    def add_zeros_alderly(string, max_len=5):
        len_str = len(string)
        for i in range(max_len - len_str):
            string = '0' + string
        return string

    @staticmethod
    def __load_alderly(folder_path):
        dataset = []
        frame_matches = pd.read_csv(folder_path + 'framematches.csv')
        for value in frame_matches.values:
            path_a = folder_path + 'FRAMESA/Image' + DataLoader.add_zeros_alderly(str(value[0])) + '.jpg'
            path_b = folder_path + 'FRAMESB/Image' + DataLoader.add_zeros_alderly(str(value[1])) + '.jpg'
            dataset.append([path_a, path_b, 1, None])
        return dataset


    @staticmethod
    def __load_kitti(folder_path):
        dataset = []
        print 'processing kitti dataset....'
        folder_path_2k11 = folder_path + '2011_09_26/'
        sequence_folders = osh.list_dir(folder_path_2k11)
        images_02 = []
        for sequence_folder in sequence_folders:
            full_path = folder_path_2k11 + sequence_folder + '/image_02/data/'
            images_02 = [full_path + im_file for im_file in osh.list_dir(full_path)]
            for image_02 in images_02:
                gps_file = image_02.replace('image_02', 'oxts').replace('.png', '.txt')
                with open(gps_file) as gps_file_handle:
                    gps_coordinates = np.array(gps_file_handle.readline().split(' ')[:2], np.float64)
                dataset.append([image_02, image_02.replace('image_02', 'image_03'), 1, gps_coordinates])
        return dataset

    @staticmethod
    def floor_ceil_int(x):
        remainder = x % 10
        if remainder < 5:
            x -= remainder
        else:
            x += (10 - remainder)
        return x

    @staticmethod
    def __load_nordland(folder_path):
        ignore = range(26417, 26538)
        ignore.extend(range(28980, 29088))
        ignore.extend(range(30442, 30494))
        ignore.extend(range(31089, 31182))
        ignore.extend(range(32397, 33041))
        ignore.extend(range(35609, 35653))
        dataset = []
        print 'Processing nordland data.....'
        offset = 171
        files_nordland = [folder_path + 'summer/' + im + '.png'
                          for im in
                          sorted([im[:-4] for im in
                                  osh.list_dir(folder_path + 'summer/')], key=int)]
        gps_file = pd.read_csv(folder_path + 'nordlandsbanen_time_pos.csv')

        for im_file in files_nordland:
            im_file_num, _ = osh.split_file_extension(osh.extract_name_from_path(im_file))
            im_file_num = int(im_file_num)
            if im_file_num in ignore:
                continue
            im_file_index = DataLoader.floor_ceil_int(im_file_num - offset) / 10
            summer_image = im_file
            winter_image = im_file.replace('summer', 'winter')
            fall_image = im_file.replace('summer', 'fall')
            spring_image = im_file.replace('summer', 'spring')

            gps_coordinates = np.array(gps_file.values[im_file_index, 1:], np.float64)
            dataset.append([summer_image, winter_image, 1, gps_coordinates])
            dataset.append([summer_image, spring_image, 1, gps_coordinates])
            dataset.append([fall_image, summer_image, 1, gps_coordinates])
            dataset.append([winter_image, spring_image, 1, gps_coordinates])
            dataset.append([fall_image, winter_image, 1, gps_coordinates])
            dataset.append([spring_image, fall_image, 1, gps_coordinates])
        return dataset
