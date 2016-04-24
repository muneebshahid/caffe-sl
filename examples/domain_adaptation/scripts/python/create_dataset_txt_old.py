import os_helper as osh
import sys
import numpy as np
import random
import cv2
import copy
from itertools import izip_longest


mich_ignore = range(1264, 1272)
mich_ignore.extend(range(1473, 1524))
mich_ignore.extend(range(1553, 1565))
mich_ignore.extend(range(1623, 1628))


def get_train_test_split_len(examples_len, split):
    # put first split% of the data for training and the rest for testing
    return int(np.ceil(split * examples_len)), -int(np.floor((1 - split) * examples_len))


def split_source_target(data_set, sources, targets, label_data_limit):
    if len(sources) == 0 or len(targets) == 0:
        print "Source or target cannot be empty"
        sys.exit()
    data_set_source = []
    data_set_target = []
    data_set_test = []
    for source in sources:
        data_set_source.extend(data_set[source][0])
        data_set_source.extend(data_set[source][1])

    for target in targets:
        # negative examples
        data_set_target.extend(data_set[target][1])
        data_set_test.extend(data_set[target][1])

        label_data_indices = []
        pos_target_data = copy.deepcopy(data_set[target][0])
        # semi supervised
        if label_data_limit > 0:
            while len(label_data_indices) < label_data_limit:
                index = np.random.randint(0, len(pos_target_data) - 1)
                if index in label_data_indices:
                    continue
                label_data_indices.append(index)
                data_set_source.append(pos_target_data[index])

            for i, instance in enumerate(pos_target_data):
                if i in label_data_indices:
                    continue
                data_set_test.append(instance)
        # unsupervised
        else:
            data_set_test.extend(pos_target_data)

        while len(pos_target_data) > 1:
            index = np.random.randint(0, len(pos_target_data) - 1)
            pos_ins_1 = pos_target_data.pop(index)
            if len(pos_target_data) - 1 > 0:
                index = np.random.randint(0, len(pos_target_data) - 1)
            else:
                index = 0
            pos_ins_2 = pos_target_data.pop(index)

            pos_ins_3 = [pos_ins_1[0], pos_ins_2[1]]
            random.shuffle(pos_ins_3)
            pos_ins_3.extend([pos_ins_1[2], pos_ins_1[3]])
            pos_ins_4 = [pos_ins_2[0], pos_ins_1[1]]
            random.shuffle(pos_ins_4)
            pos_ins_4.extend([pos_ins_1[2], pos_ins_1[3]])
            # since we do not
            data_set_target.extend([pos_ins_3, pos_ins_4])
    # shuffling data
    random.shuffle(data_set_source)
    random.shuffle(data_set_target)
    random.shuffle(data_set_test)
    return data_set_source, data_set_target, data_set_test


def get_batch(index, batch_size, data):
    len_data = len(data)
    reached_end = False
    if index + batch_size < len_data:
        batch = data[index: index + batch_size]
        index += batch_size
    else:
        batch = data[index:]
        # randint: len_source - 1
        rand_indices = np.random.randint(0, len_data, batch_size - len(batch))
        batch.extend([data[index] for index in rand_indices])
        reached_end = True
    return batch, reached_end


def pad_train_data(source, target, batch_size=320):
    half_batch_size = batch_size / 2
    len_source = len(source)
    len_target = len(target)
    print "source len: {0}, target len {1}, batch {2}".format(len_source, len_target, batch_size)
    # half of the data set in each batch should be test data
    padded = []
    end = False
    s_index = 0
    t_index = 0

    while not end:
        s_batch, s_reached_end = get_batch(s_index, half_batch_size, source)
        assert len(s_batch) == half_batch_size
        t_batch, t_reached_end = get_batch(t_index, half_batch_size, target)
        assert len(t_batch) == half_batch_size
        padded.extend(s_batch)
        padded.extend(t_batch)

        if t_reached_end:
            t_index = 0
            random.shuffle(target)
        else:
            t_index += half_batch_size
        s_index += half_batch_size
        end = s_reached_end
    assert len(padded) % batch_size == 0
    return padded


def split_data(train, test, pos, neg, split):
    pos_train_examples, pos_test_examples = get_train_test_split_len(len(pos), split)
    neg_train_examples, neg_test_examples = get_train_test_split_len(len(neg), split)
    # ensures we do not append the same sequence again
    random.shuffle(pos)
    random.shuffle(neg)
    for pos_example, neg_example in izip_longest(pos[0:pos_train_examples], neg[0:neg_train_examples]):
        if pos_example is not None:
            train.append(pos_example)
        if neg_example is not None:
            train.append(neg_example)
    for pos_example, neg_example in izip_longest(pos[pos_test_examples:], neg[neg_test_examples:]):
        if pos_example is not None:
            test.append(pos_example)
        if neg_example is not None:
            test.append(neg_example)
    # ensures an already processed data set does not always stay at the beginning
    random.shuffle(train)
    random.shuffle(test)


def progress(current, total):
    return '{0} / {1}'.format(current, total)


def write(data_set, file_path, file_num = None):
    if file_num is not None:
        with open(file_path + '.txt', 'w') as w:
            # file1 uses columns 0 and 2, while file2 uses columns 1 and 3
            w.writelines(
                    [str(instance[file_num - 1]).replace('\\', '/') + ' ' + str(instance[file_num + 1]) + '\n' for instance
                     in data_set])
    else:
        with open(file_path + '.txt', 'w') as w:
            # file1 uses columns 0 and 2, while file2 uses columns 1 and 3
            w.writelines([instance for instance in data_set])


def get_random_batch(data, size):
    random_batch = []
    indices = []
    while len(random_batch) < size:
        index = np.random.randint(0, len(data))
        if index not in indices:
            indices.append(index)
            random_batch.append(data[index])
    return random_batch


def pad_data_multiple(data, batch_size):
    remainder = len(data) % batch_size
    temp_data = []
    if remainder > 0:
        pad_size = batch_size - remainder
        rnd_batch = get_random_batch(data, pad_size)
        temp_data.extend(rnd_batch)
    return temp_data


def extend_data(data):
    temp_data = copy.deepcopy(data)
    random.shuffle(temp_data)
    return temp_data


def process_freiburg(data_set, source, key, root_folder_path, sub_folder):
    folder_path = root_folder_path + sub_folder
    print "processing freiburg data....."
    save_neg_im = False
    data_set_freiburg_pos = []
    data_set_freiburg_neg = []
    processed_season_match = open(folder_path + 'processed_season_match.txt', "r")
    # sorry i realize it might be quite cryptic but i could'nt help myself
    # read lines and set labels to 1 1 (similarity and domain label)
    data_set_freiburg = [array
                         for array in
                         (line.replace('uncompressed', 'freiburg/uncompressed')
                              .replace('\n', ' 1 ' + str(int(source))).split(' ')
                                for line in processed_season_match.readlines())]
    processed_season_match.close()
    for instance in data_set_freiburg:
        i = 1
        while len(instance) - 2 > i:
            seasons = [instance[0], instance[i]]
            if source:
                random.shuffle(seasons)
            seasons.extend(instance[-2:])
            data_set_freiburg_pos.append(seasons)
            i += 1
    del data_set_freiburg
    pos_examples = len(data_set_freiburg_pos)
    neg_examples = 0
    image_gap = 200
    last_index = len(data_set_freiburg_pos) - 1
    while neg_examples < pos_examples:
        im1 = np.random.randint(0, last_index)
        im_diff = im1 - image_gap - 1
        while True:
            im2 = np.random.random_integers(0, last_index)
            if abs(im1 - im2) > im_diff:
                break
        if save_neg_im:
            im_1 = cv2.imread(folder_path + data_set_freiburg_pos[im1][0])
            im_2 = cv2.imread(folder_path + data_set_freiburg_pos[im2][1])
            cv2.imwrite(folder_path + 'neg_im/' + str(neg_examples) + '.png',
                        np.concatenate((im_1, im_2), axis=1))
            print 'saving neg example {0} / {1}'.format(neg_examples, pos_examples)
        seasons = [data_set_freiburg_pos[im1][0], data_set_freiburg_pos[im2][1]]
        # since we do not deal with negative data later on so we can shuffle it here
        random.shuffle(seasons)
        seasons.extend([0, source])
        data_set_freiburg_neg.append(seasons)
        if neg_examples % 100 == 0:
            print "{0} / {1}".format(neg_examples, pos_examples)
        neg_examples += 1
    data_set[key] = [data_set_freiburg_pos, data_set_freiburg_neg]


def process_michigan(data_set, source, key, root_folder_path, sub_folder):
    folder_path = root_folder_path + sub_folder
    print 'Processing michigan data.....'
    data_set_michigan_pos = []
    data_set_michigan_neg = []
    # Process Michigan data
    months_mich = ['aug', 'jan']
    files_michigan_pos = [im.replace('\\', '/')
                          for im in sorted(osh.get_folder_contents(folder_path + 'jan/', '*.tiff'))]

    print 'Creating positive examples'
    for jan_file in files_michigan_pos:
        file_n = osh.extract_name_from_path(jan_file)

        if int(file_n[5:-5]) in mich_ignore:
            print "ignoring {0}".format(file_n[5:-5])
            continue
        jan_file = jan_file.replace(root_folder_path, '')
        if source:
            month_mich = np.random.random_integers(0, 1)
        else:
            month_mich = 0
        path_1 = jan_file.replace('jan', months_mich[month_mich])
        path_2 = jan_file.replace('jan', months_mich[abs(month_mich - 1)])
        # label_1 is similarity label, while label_2 is the domain_label
        data_set_michigan_pos.append([path_1, path_2, 1, source])

    del files_michigan_pos
    images_gap = 600
    michigan_neg_instances = 0
    michigan_pos_len = len(data_set_michigan_pos)
    last_index = michigan_pos_len - 1
    print "found {0} positive examples".format(michigan_pos_len)
    print 'Creating negative examples'
    while michigan_neg_instances < michigan_pos_len:
        im1 = np.random.random_integers(0, last_index)
        # since we do not deal with negative data later on so we can shuffle it here
        month_mich = np.random.random_integers(0, 1)
        file_1 = folder_path + months_mich[month_mich] + '/00000' + str(im1) + '.tiff'
        # michigan data set has a weird naming convention, so checking if the file actually exists
        if not osh.is_file(file_1):
            continue
        # ensure image gap between negative examples
        im_diff = im1 - images_gap
        while True:
            im2 = np.random.random_integers(0, last_index)
            if abs(im1 - im2) > im_diff:
                break
        file_2 = folder_path + months_mich[abs(month_mich - 1)] + '/00000' + str(im2) + '.tiff'
        if not osh.is_file(file_2):
            continue
        file_1 = file_1.replace(folder_path, 'michigan/uncompressed/')
        file_2 = file_2.replace(folder_path, 'michigan/uncompressed/')
        data_set_michigan_neg.append([file_1, file_2, 0, source])
        michigan_neg_instances += 1
        if michigan_neg_instances % 2000 == 0:
            print 'negative examples: ', progress(michigan_neg_instances, michigan_pos_len)

    print "created {0} negative examples".format(len(data_set_michigan_neg))
    data_set[key] = [data_set_michigan_pos, data_set_michigan_neg]


def get_fukui_im_path(image_id, gt_id, root_folder, is_query=False):
    path = root_folder + ('query/' if is_query else 'db/') + gt_id + image_id
    return path


def return_example(im1, im2, similarity_label, root_folder_path, is_source):
    example = [im1.replace(root_folder_path, ''), im2.replace(root_folder_path, '')]
    if is_source:
        random.shuffle(example)
    example.extend([similarity_label, is_source])
    return example


def process_fukui(data_set, source, key, root_folder_path, sub_folder):
    print 'processing fukui data.....'
    extra_im_range = 3
    seasons = ['AU', 'SP', 'SU', 'WI']
    data_set_fukui_pos = []
    data_set_fukui_neg = []
    folder_path = root_folder_path + sub_folder

    # we need data set for each season
    print 'creating examples'
    for season in seasons:
        data_set_fukui_season_pos = []
        data_set_fukui_season_neg = []
        print 'processing: ', season
        season_folder = folder_path + season + '/'
        ground_truth_folder = season_folder + 'gt/'
        gts = osh.list_dir(ground_truth_folder)
        print 'creating positive examples'
        for gt in gts:
            gt_id = gt[:-4]
            with open(ground_truth_folder + gt, "r") as ground_truths:
                for line in ground_truths.readlines():
                    # skip \n
                    images = line.replace('\n', '').split(' ')
                    qu_image = images[0]
                    db_image = images[1]
                    qu_image_path = get_fukui_im_path(qu_image + '.jpg', gt_id + '/', season_folder, True)
                    db_image_path = get_fukui_im_path(db_image + '.jpg', gt_id + '/', season_folder)
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
                    gt_example = return_example(qu_image_path, db_image_path, 1, root_folder_path, source)

                    data_set_fukui_season_pos.append(gt_example)
                    int_db_image = int(db_image)
                    im_range = range(int(images[2]), int(images[3]) + 1)

                    # append images before if possible
                    for im in range(int_db_image - extra_im_range, int_db_image):
                        if im in im_range:
                            db_image_path = get_fukui_im_path('0' + str(im) + '.jpg', gt_id + '/', season_folder)
                            if osh.is_file(db_image_path):
                                gt_example = return_example(qu_image_path, db_image_path, 1, root_folder_path, source)
                                data_set_fukui_season_pos.append(gt_example)

                    # append images after if possible
                    for im in range(int_db_image + 1, int_db_image + extra_im_range + 1):
                        if im in im_range:
                            db_image_path = get_fukui_im_path('0' + str(im) + '.jpg', gt_id + '/', season_folder)
                            if osh.is_file(db_image_path):
                                gt_example = return_example(qu_image_path, db_image_path, 1, root_folder_path, source)
                                data_set_fukui_season_pos.append(gt_example)

        len_season_pos = len(data_set_fukui_season_pos)
        print 'positive examples created: ', len_season_pos
        print 'creating negative examples'
        pos_groups = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]

        while len(data_set_fukui_season_neg) < len_season_pos:
            f1, f2 = np.random.random_integers(1, 12, 2)
            repeat = False
            for group in pos_groups:
                if f1 in group and f2 in group:
                    repeat = True
            if repeat:
                continue
            path_1 = season_folder + ('query/' if season == 'WI' else 'db/') + str(f1) + '/'
            path_2 = season_folder + 'db/' + str(f2) + '/'
            files_1 = osh.list_dir(path_1)
            files_2 = osh.list_dir(path_2)
            im_1 = np.random.randint(0, len(files_1))
            im_2 = np.random.randint(0, len(files_2))
            path_1 += files_1[im_1]
            path_2 += files_2[im_2]
            if osh.is_file(path_1) and osh.is_file(path_2):
                gt_example = return_example(path_1, path_2, 0, root_folder_path, source)
                data_set_fukui_season_neg.append(gt_example)
        print 'negative examples created: ', len(data_set_fukui_season_neg)

        data_set_fukui_pos.extend(data_set_fukui_season_pos)
        data_set_fukui_neg.extend(data_set_fukui_season_neg)

    print 'total positive examples: ', len(data_set_fukui_pos)
    print 'total negative examples: ', len(data_set_fukui_neg)
    data_set[key] = [data_set_fukui_pos, data_set_fukui_neg]


def main(label_data_limit=0):
    root_folder_path = osh.get_env_var('CAFFE_ROOT') + '/../data/images/' + '/'
    root_folder_path = root_folder_path.replace('\\', '/')
    if not osh.is_dir(root_folder_path):
        print "source folder does'nt exist, existing....."
        sys.exit()

    # batch size is used for padding.
    batch_size = 128 

    # flag to pad source and target arrays to make them a multiple of batch size
    pad_multiple = True

    # flag to pad train data (source) with target
    pad_train = True

    create_mean_data = False
    # until a custom shuffling is implementd in the data layer, 
    # pseudo shuffle the data by extending it with random repititons
    # of the whole data set
    pseudo_shuffle = 5
    source_mich = True
    source_freiburg = False
    source_fukui = True
    source = []
    target = []
    source_data = []
    target_data = []
    data_set = {}
    if source_mich is not None:
        key = 'michigan'
        if source_mich:
            source.append(key)
        else:
            target.append(key)
        process_michigan(data_set, int(source_mich), key, root_folder_path, 'michigan/uncompressed/')
    if source_freiburg is not None:
        key = 'freiburg'
        if source_freiburg:
            source.append(key)
        else:
            target.append(key)
        process_freiburg(data_set, int(source_freiburg), key, root_folder_path, 'freiburg/')
    if source_fukui:
        key = 'fukui'
        if source_fukui:
            source.append(key)
        else:
            target.append(key)
        process_fukui(data_set, int(source_fukui), key, root_folder_path, 'fukui/')

    print "splitting in to target and source"
    source_data_orig, target_data_orig, test_data = split_source_target(data_set, source, target, label_data_limit)
    source_data.extend(source_data_orig)
    target_data.extend(target_data_orig)
    print "source size {0} target size {1} test {2}".format(len(source_data), len(target_data), len(test_data))

    i = 1
    while i < pseudo_shuffle:
        print "extending data {0} time".format(i)
        source_data.extend(extend_data(source_data_orig))
        # target_data.extend(extend_data(target_data_orig))
        i += 1
    print "extended len: source {0} target {1} test {2}".format(len(source_data), len(target_data), len(test_data))

    if pad_multiple:
        print "padding data to nearest multiple of batch size"
        source_data.extend(pad_data_multiple(source_data, batch_size))
        target_data.extend(pad_data_multiple(target_data, batch_size))
        print "padded source size {0} target size {1} test {2}".format(len(source_data), len(target_data), len(test_data))

    print "padded len source {0} target size {1} test {2}".format(len(source_data), len(target_data), len(test_data))

    if pad_train:
        print "padding source data with target data"
        padded = pad_train_data(copy.deepcopy(source_data_orig), copy.deepcopy(target_data_orig), batch_size)

    print "writing data files"

    write(source_data, root_folder_path + 'source1', 1)
    write(source_data, root_folder_path + 'source2', 2)
    write(target_data, root_folder_path + 'target1', 1)
    write(target_data, root_folder_path + 'target2', 2)
    write(test_data, root_folder_path + 'test1', 1)
    write(test_data, root_folder_path + 'test2', 2)

    if pad_train:
        write(padded, root_folder_path + 'train1', 1)
        write(padded, root_folder_path + 'train2', 2)
    if create_mean_data:
        source_target_data_set = []
        print "creating data set for image mean"
        for domain in data_set:
            instances = data_set[domain]
            for pos, neg in izip_longest(instances[0], instances[1]):
                pos_neg = [pos, neg]
                for instance in pos_neg:
                    if instance is not None:
                        for im in instance[:-2]:
                            im += ' 1\n'
                            if im not in source_target_data_set:
                                source_target_data_set.append(im)
        print "writing data set for image mean"
        write(source_target_data_set, root_folder_path + 'complete')

if __name__ == "__main__":    
    main()
'''
#Process Amos data
valid_camera_nums = np.load('amos/valid_camera_nums.npy')
if not osh.path_exists(folder_path_amos):
	print "Invalid source folder path"

folder_contents = osh.get_folder_contents(folder_path_amos)
data_set = []
limit_per_folder = 10000
for folder_content in folder_contents:	
	if osh.is_dir(folder_content) and osh.extract_name_from_path(folder_content) in valid_camera_nums:
		camera_folder_path = folder_content + '/'
		camera_num = osh.extract_name_from_path(folder_content)		
		print "Found Amos Camera #: " + camera_num
		camera_folder_contents = osh.get_folder_contents(camera_folder_path)
		months_per_cam = len(camera_folder_contents)
		print "Found Months: " + str(months_per_cam)
		im_added = 0
		if months_per_cam == 1:
			month = camera_folder_contents[0] + '/'
			print "In month: " + month
			images = osh.get_folder_contents(month, '*.jpg')
			total_images = len(images)
			print "Images Found: " + str(total_images)
			while im_added < limit_per_folder:				
				im1, im2 = np.random.random_integers(0, total_images - 1, 2)
				data_set.append([im1, im2, 1])
				im_added = im_added + 1
				if im_added % 500 == 0:
					print 'Added: ' + str(im_added) + ' / ' + str(limit_per_folder)
			break					
'''
