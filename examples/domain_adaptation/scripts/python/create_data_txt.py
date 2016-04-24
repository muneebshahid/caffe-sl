import os_helper as osh
import sys
import numpy as np
import random
import copy
from data_loader import DataLoader


def get_train_test_split_len(examples_len, split):
    # put first split% of the data for training and the rest for testing
    return int(np.ceil(split * examples_len)), -int(np.floor((1 - split) * examples_len))


def split_train_test(data_set, split=0.7):
    train_examples, test_examples = get_train_test_split_len(len(data_set), split)
    # ensures we do not append the same sequence again
    return data_set[0:train_examples], data_set[train_examples:]


def shuffle_columns(instance):
    return [instance[1], instance[0], instance[3], instance[2]]


def get_fukui_im_path(image_id, gt_id, root_folder, is_query=False):
    path = root_folder + ('query/' if is_query else 'db/') + gt_id + image_id
    return path


def extend_data(data):
    temp_data = copy.deepcopy(data)
    random.shuffle(temp_data)
    return temp_data


def print_progress(curr_iterate, total_iteration, print_after_iterations=50000):
    if curr_iterate % print_after_iterations == 0:
        print "{0} / {1}".format(curr_iterate, total_iteration)


def get_augmented_data_pos(data_set, ext, limit=4):
    augmented_data = []
    augmented_dict = dict()
    ext_len = len(ext)
    for instance in data_set:
        for im in instance[:2]:
            if im not in augmented_dict:
                augmented_dict[im] = []
        im_1 = instance[0][:-ext_len]
        im_2 = instance[1][:-ext_len]
        keys_dict_1 = {key: [str(rnd) for rnd in random.sample(range(0, 4), limit)] for key in AUGMENTED_KEYS}
        keys_dict_2 = {key: [str(rnd) for rnd in random.sample(range(0, 4), limit)] for key in AUGMENTED_KEYS}

        while len(keys_dict_1) > 0 and len(keys_dict_2) > 0:
            # get actual keys
            key_1, key_2 = random.choice(keys_dict_1.keys()), random.choice(keys_dict_2.keys())

            # pop the elements at the corresponding indices
            aug_im_1, aug_im_2 = keys_dict_1[key_1].pop(), keys_dict_2[key_2].pop()

            # create actual image names
            id_1 = im_1.replace('orig', 'augmented') + '-' + key_1 + aug_im_1 + ext
            id_2 = im_2.replace('orig', 'augmented') + '-' + key_2 + aug_im_2 + ext

            # append to data
            augmented_data.append([instance[0], id_2, 1, instance[-1]])
            augmented_data.append([id_1, instance[1], 1, instance[-1]])
            augmented_data.append([id_1, id_2, 1, instance[-1]])

            # add to dict
            augmented_dict[instance[0]].append(id_1)
            augmented_dict[instance[1]].append(id_2)

            # check if end of dict vars
            if len(keys_dict_1[key_1]) == 0:
                keys_dict_1.pop(key_1)
            if len(keys_dict_2[key_2]) == 0:
                keys_dict_2.pop(key_2)
    return augmented_data, augmented_dict


def get_distant_images_fukui(dataset, image_gap, im1_fixed_index=None):
    im1, im2 = [], []
    pos_groups = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
    g1, g2 = random.sample(pos_groups, 2)

    im1 = random.choice(dataset) if im1_fixed_index is None else dataset[im1_fixed_index]
    im_1_num = int(im1[0].replace('\\', '/').split('/')[-2])
    repeat = True
    while repeat:
        im2 = random.choice(dataset)
        im_2_num = int(im2[0].replace('\\', '/').split('/')[-2])
        for group in pos_groups:
            repeat = False
            if im_1_num in group and im_2_num in group:
                repeat = True
                break
    return im1, im2


def get_distant_images(dataset, image_gap, gps_gap, is_fukui, im1_fixed_index=None):
    if is_fukui:
        return get_distant_images_fukui(dataset, image_gap, im1_fixed_index)
    data_len = len(dataset)
    im_1, im_2 = None, None
    # gps data available
    if dataset[0][-1] is not None and gps_gap != 0:
        while True:
            if im1_fixed_index is None:
                im_1, im_2 = random.sample(dataset, 2)
            else:
                im_1 = dataset[im1_fixed_index]
                im_2 = random.choice(dataset)
            if np.linalg.norm(im_1[-1] - im_2[-1]) > gps_gap:
                break
    else:
        im_index_1 = np.random.randint(0, data_len) if im1_fixed_index is None else im1_fixed_index
        im_diff = im_index_1 - image_gap - 1
        while True:
            im_index_2 = np.random.randint(0, data_len)
            if abs(im_index_1 - im_index_2) > im_diff:
                break
        # else:
        #
        #     if im_index_1 + image_gap >= data_len:
        #         im_index_2 = im_index_1 - image_gap
        #     else:
        #         im_index_2 = im_index_1 + image_gap
        im_1, im_2 = dataset[im_index_1], dataset[im_index_2]
    return im_1, im_2


def evenly_mix_source_target(dataset, batch_size=8):
    i = 0
    while True:
        if i <= len(dataset) - batch_size:
            batch = dataset[i:i + batch_size]
            for j, instance in enumerate(batch[0:len(batch)/2]):
                batch[j] = shuffle_columns(instance)
            random.shuffle(batch)
            dataset[i:i + batch_size] = batch
        else:
            batch = dataset[i:]
            for j, instance in enumerate(batch[0:len(batch)/2]):
                batch[j] = shuffle_columns(instance)
            random.shuffle(batch)
            dataset[i:] = batch
            break
        i += batch_size


def get_im_distances(key):
    image_gap, gps_gap = 0, 0
    if key == 'freiburg':
        image_gap = 200
        gps_gap = .00045
    elif key == 'michigan':
        image_gap = 600
    elif key == 'fukui':
        image_gap = 170
    elif key == 'alderly':
        image_gap = 100
    elif key == 'kitti':
        image_gap = 100
        gps_gap = .00045
    elif key == 'nordland':
        gps_gap = .00045
    return image_gap, gps_gap


def create_negatives(key, dataset, length=None, augmented=False, chosen_aug=None, select_orig=0.7):
    print 'processing {0} negative images: '.format(key)
    negatives = []
    pos_examples = len(dataset) if length is None else length
    neg_examples = 0
    image_gap, gps_gap = get_im_distances(key)
    assert image_gap > 0 or gps_gap > 0
    while neg_examples < pos_examples:
        ims = get_distant_images(dataset, image_gap, gps_gap, is_fukui=key == 'fukui')
        if not augmented:
            negatives.append([ims[0][0], ims[1][1], 0, ims[0][-1]])
        else:
            instance = []
            likelihoods = np.random.rand(2, 1)
            for i, likelihood in enumerate(likelihoods):
                if likelihood < select_orig:
                    instance.append(ims[i][i])
                elif chosen_aug is not None:
                        instance.append(random.choice(chosen_aug[ims[i][i]]))
                else:
                    im_file = []
                    im_file.extend(osh.split_file_extension(ims[0][0]))
                    im_file.extend(osh.split_file_extension(ims[1][1]))
                    f_name, f_ext = im_file[i]
                    aug_key = random.choice(AUGMENTED_KEYS)
                    num = random.choice(range(0, 3))
                    full_im = f_name + '-' + aug_key + str(num) + f_ext
                    instance.append(full_im)
            instance.extend([0, ims[0][-1]])
            negatives.append(instance)
        print_progress(neg_examples, pos_examples)
        neg_examples += 1
    return negatives


def write_data(data_set, root_folder_path, write_path, file_path, file_num=None, lmdb=True):
    file_path = write_path + '/' + file_path
    if file_num is not None:
        with open(file_path + '.txt', 'w') as w:
            # file1 uses columns 0 and 2, while file2 uses columns 1 and 3
            w.writelines([str(instance[file_num - 1]).replace('\\', '/') + ' ' +
                          str(instance[-2]) +
                          '\n' for instance in data_set])
    else:
        with open(file_path + '.txt', 'w') as w:
           w.writelines(
                    [instance[0].replace('\\', '/') + ' ' + str(instance[1]) + '\n' for instance
                     in data_set])


def evenly_mix_pos_neg(data_pos, data_neg, batch_size=8):
    data = []
    i = 0
    j = 0
    # assumes neg >= pos
    assert len(data_pos) <= len(data_neg)
    batch_half = batch_size / 2
    run = True
    while run:
        if i <= len(data_neg) - batch_half:
            batch_neg = data_neg[i:i + batch_half]
            i += batch_half
        else:
            batch_neg = data_neg[i:]
            run = False
        if j <= len(data_pos) - batch_half:
            batch_pos = data_pos[j:j + batch_half]
            j += batch_half
        else:
            batch_pos = data_pos[j:]
            j = 0
            random.shuffle(data_pos)
        batch = batch_pos
        batch.extend(batch_neg)
        random.shuffle(batch)
        data.extend(batch)
    return data


def pseudo_shuffle_data(data, pseudo_shuffle):
    if pseudo_shuffle > 0:
        data_orig = copy.deepcopy(data)
        i = 0
        while i < pseudo_shuffle:
            print "extending train data {0} time".format(i + 1)
            data.extend(extend_data(data_orig))
            i += 1


def create_triplets_data(key, data, triplets_dim):
    triplet_data = []
    num_triplets, size_triplets = triplets_dim
    image_gap, gps_gap = get_im_distances(key)
    for i, pos_data in enumerate(data):
        image_1 = i
        for j in range(num_triplets):
            instance = pos_data[:2]
            negatives = []
            for k in range(size_triplets):
                _, im2 = get_distant_images(dataset=data, image_gap=image_gap, gps_gap=gps_gap, is_fukui=key == 'fukui',
                                            im1_fixed_index=image_1)
                negatives.append(im2[1])
            instance.extend(negatives)
            instance.append(pos_data[-1])
            triplet_data.append(instance)
    print '{0} data len: {1}'.format(key, len(triplet_data))
    return triplet_data


def flatten_triplets(data):
    flattened_data = []
    for multi_dim_data in data:
        for i, paired_image in enumerate(multi_dim_data[:-1]):
            flattened_data.append([paired_image, 1 if i <= 1 else 0])
    return flattened_data


def pad_triplets(data, triplet_size, batch_size=640):
    data_len = len(data)
    remainder = data_len % float(batch_size)
    if remainder > 0:
        final_size = data_len - remainder + batch_size
        triples_needed = (final_size - data_len) / triplet_size
        indices = np.random.randint(0, data_len/4.0, triples_needed)
        for index in indices:
            for triplet in range(triplet_size):
                final_index = triplet_size * index + triplet
                data.append(data[final_index])


def process_datasets(keys, root_folder_path, write_path, augmented_limit, neg_limit, triplets_dim=None):
    train_data_triplets = []
    test_data_triplets = []
    train_data_aug_pos = []
    train_data_aug_neg = []
    train_data_pos = []
    train_data_neg = []
    test_data_pos = []
    test_data_neg = []
    for key in keys:
        data_set_pos = DataLoader.load(key, root_folder_path)
        # Add fukui data only for training.
        if key != 'fukui':
            train_data_pos_temp, test_data_pos_temp = split_train_test(data_set_pos)
            if triplets_dim is not None:
                train_data_temp = create_triplets_data(key, train_data_pos_temp, triplets_dim[key])
                test_data_temp = create_triplets_data(key, test_data_pos_temp, [1, 1])
            else:
                train_data_neg_temp = create_negatives(key, train_data_pos_temp, length=neg_limit[key])
                test_data_neg_temp = create_negatives(key, test_data_pos_temp, length=len(test_data_pos_temp))
        else:
            train_data_pos_temp, test_data_pos_temp = data_set_pos, []
            if triplets_dim is not None:
                train_data_temp = create_triplets_data(key, train_data_pos_temp, triplets_dim[key])
                test_data_temp = []
            else:
                train_data_neg_temp = create_negatives(key, train_data_pos_temp, length=neg_limit[key])
                test_data_neg_temp = []

        if triplets_dim:
            train_data_triplets.extend(train_data_temp)
            test_data_triplets.extend(test_data_temp)
        else:
            train_data_pos.extend(train_data_pos_temp)
            train_data_neg.extend(train_data_neg_temp)
            test_data_pos.extend(test_data_pos_temp)
            test_data_neg.extend(test_data_neg_temp)

            if augmented_limit is not None:
                train_data_aug_pos.extend(train_data_pos_temp)
                train_data_aug_neg.extend(train_data_neg_temp)
                augmented_pos, augmented_dict = get_augmented_data_pos(train_data_pos_temp, '.jpg', augmented_limit[key])
                augmented_neg = create_negatives(key, train_data_pos_temp,
                                                 len(augmented_pos), True,
                                                 augmented_dict, 0.7)
                train_data_aug_pos.extend(augmented_pos)
                train_data_aug_neg.extend(augmented_neg)

    if triplets_dim is not None:
        random.shuffle(train_data_triplets)
        print "triplet data {0}".format(len(train_data_triplets))
        pseudo_shuffle_data(data=train_data_triplets, pseudo_shuffle=0)
        print "extended triplet data {0}".format(len(train_data_triplets))
        print "triplet test data {0}".format(len(test_data_triplets))
        flattened_triplets_train = flatten_triplets(train_data_triplets)
        flattened_triplets_test = flatten_triplets(test_data_triplets)
        print "flattened triplet data train {0}".format(len(flattened_triplets_train))
        print "flattened triplet data test {0}".format(len(flattened_triplets_test))
        pad_triplets(flattened_triplets_train, 4, 640)
        print "padded triplet data train {0}".format(len(flattened_triplets_train))
        print "writing files...."
        write_data(flattened_triplets_train, root_folder_path, write_path, 'triplet_data_train')
        write_data(flattened_triplets_test, root_folder_path, write_path, 'triplet_data_test')
    else:
        random.shuffle(train_data_pos)
        random.shuffle(train_data_neg)

        print "train data pos {0}, train data neg {1}".format(len(train_data_pos), len(train_data_neg))
        pseudo_shuffle_data(data=train_data_pos, pseudo_shuffle=0)
        pseudo_shuffle_data(data=train_data_neg, pseudo_shuffle=0)
        print "extended train data pos {0}, train data neg {1}".format(len(train_data_pos), len(train_data_neg))

        train_data = evenly_mix_pos_neg(data_pos=train_data_pos,
                                        data_neg=train_data_neg,
                                        batch_size=8)
        test_data = test_data_pos
        test_data.extend(test_data_neg)

        print "train data {0}".format(len(train_data))
        print "test data {0}".format(len(test_data))
        print 'writing files....'
        write_data(train_data, root_folder_path, write_path, 'train1', file_num=1, lmdb=False)
        write_data(train_data, root_folder_path, write_path, 'train2', file_num=2, lmdb=False)
        write_data(test_data, root_folder_path, write_path, 'test1', file_num=1, lmdb=False)
        write_data(test_data, root_folder_path, write_path, 'test2', file_num=2, lmdb=False)

        if augmented_limit is not None:
            random.shuffle(train_data_aug_pos)
            random.shuffle(train_data_aug_neg)
            print "augmented train data pos {0}, train data neg {1}".format(len(train_data_aug_pos),
                                                                            len(train_data_aug_neg))
            pseudo_shuffle_data(data=train_data_aug_pos, pseudo_shuffle=0)
            pseudo_shuffle_data(data=train_data_aug_neg, pseudo_shuffle=0)
            print "augmented extended train data pos {0}, train data neg {1}".format(len(train_data_aug_pos),
                                                                                     len(train_data_aug_neg))
            train_data_aug = evenly_mix_pos_neg(data_pos=train_data_aug_pos,
                                                data_neg=train_data_aug_neg,
                                                batch_size=8)
            print "augmented train data {0}".format(len(train_data_aug))
            print "writing files...."
            write_data(train_data_aug, root_folder_path, write_path, 'train_aug_1', file_num=1, lmdb=False)
            write_data(train_data_aug, root_folder_path, write_path, 'train_aug_2', file_num=2, lmdb=False)


def main():
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    print caffe_root
    root_folder_path = caffe_root + '/../data/images/orig/'
    root_folder_path = root_folder_path.replace('\\', '/')
    if not osh.is_dir(root_folder_path):
        print "source folder does'nt exist, existing....."
        sys.exit()
    keys = ['freiburg', 'michigan', 'nordland', 'fukui', 'kitti', 'alderly']
    write_path = caffe_root + '/data/domain_adaptation_data/images/'
    augmented_limit = {keys[0]: 1, keys[1]: 1, keys[2]: 1, keys[3]: 1, keys[4]: 1, keys[5]: 1}
    neg_limit = {keys[0]: None,
                 keys[1]: 300000,
                 keys[2]: 300000,
                 keys[3]: 100000,
                 keys[4]: 300000,
                 keys[5]: 300000}
    triplet_limit = {keys[0]: [20, 2],
                     keys[1]: [12, 2],
                     keys[2]: [2, 2],
                     keys[3]: [10, 2],
                     keys[4]: [2, 2],
                     keys[5]: [10, 2],}
    process_datasets(keys, root_folder_path, write_path, None, neg_limit, None)

if __name__ == "__main__":
    AUGMENTED_KEYS = ['tra' , 'rot', 'aff', 'per']
    main()
