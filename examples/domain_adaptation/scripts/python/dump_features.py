from extract_features import FeatureExtractor
import numpy as np
import os_helper as osh


def load_file(path, keys):
    data = {}
    for key in keys:
        data[key] = []
        with open(path + '/test1.txt') as test_1, open(path + '/test2.txt') as test_2:
            for line_1, line_2 in zip(test_1.readlines(), test_2.readlines()):
                line_1 = line_1.replace('\n', '').split(' ')
                line_2 = line_2.replace('\n', '').split(' ')
                if line_1[1] == '1' and line_2[1] == '1':
                    if key in line_1[0] and key in line_2[0]:
                        data[key].append([line_1[0], line_2[0]])
    return data


def dump_results(model_folder, model_id, key, sub_key, feature_key, results):
    np.save(save_path + model_folder + '_' + model_id + '_' + key + '_' + sub_key + '_' + feature_key, results)


def normalize(feature):
    return feature / np.linalg.norm(feature)


def filter_data(key, sub_key, dataset):
    filtered_images = []
    if key == 'nordland':
        ignore = range(26417, 26538)
        ignore.extend(range(28980, 29088))
        ignore.extend(range(30442, 30494))
        ignore.extend(range(31089, 31182))
        ignore.extend(range(32397, 33041))
        ignore.extend(range(35609, 35653))
        for pair in dataset:
            if ('summer' in pair[0] or 'summer' in pair[1]) and ('winter' in pair[0] or 'winter' in pair[1]):
                file_id, _ = osh.split_file_extension(osh.extract_name_from_path(pair[0]))
                if int(file_id) in ignore:
                    continue
                if 'summer' in pair[0]:
                    image = pair[0].replace('summer', sub_key)
                else:
                    image = pair[1].replace('summer', sub_key)
                filtered_images.append(image)
    elif key == 'freiburg':
        for pair in dataset:
            if sub_key == 'summer':
                filtered_images.append(pair[0])
            else:
                filtered_images.append(pair[1])
    else:
        filtered_images = dataset
    return filtered_images


def main():
    fe = FeatureExtractor(model_path=caffe_model_path, deploy_path=deploy_path, mean_binary_path=mean_binary_path,
                          input_layer=input_layers)
    #data_set_keys = {'nordland': ['summer', 'winter', 'spring', 'fall']}#'michigan'i, 'freiburg' ]
    data_set_keys = {'freiburg': ['summer', 'winter'] }
    data = load_file(txt_path, data_set_keys.keys())
    model_id = osh.extract_name_from_path(caffe_model_path)
    for key in data:
        print 'processing: {0}_{1}_{2}'.format(model_folder, model_id, key)
        for sub_key in data_set_keys[key]:
            print 'processing: {0}'.format(sub_key)
            # for each feature we dump two features normalized and un normalized
            features = [[[], []] for feature_layer in feature_layers]
            key_data = filter_data(key, sub_key, data[key])
            key_data_len = len(key_data)
            processed = 0
            fe.set_batch_dim(batch_size, 3, 227, 227)
            print 'total data {0}'.format(key_data_len)
            num_iter = int(np.ceil(key_data_len / float(batch_size)))
            for i in range(num_iter):
                if (batch_size * (i + 1)) <= key_data_len:
                    curr_batch_size = batch_size
                else:
                    curr_batch_size = key_data_len - batch_size * i
                    fe.set_batch_dim(curr_batch_size)
                '''result = {'conv3': np.ones((curr_batch_size, 600)) * 5,
                          'conv3_p': np.random.rand(curr_batch_size, 600),
                          'fc8_n': np.random.rand(curr_batch_size, 128),
                          'fc8_n_p': np.random.rand(curr_batch_size, 128)}'''
                start_index = i * batch_size
                end_index = start_index + batch_size
                images = key_data[start_index:end_index]
                result = fe.extract(images=images,
                                    blob_keys=feature_layers)
                for i, feature_layer in enumerate(feature_layers):
                    features[i][0].extend([feature.flatten().astype(dtype=np.float64) for feature in result[feature_layer].copy()])
                    features[i][1].extend([normalize(feature.flatten().astype(dtype=np.float64)) for feature in result[feature_layer].copy()])
                processed += curr_batch_size
                print '{0} / {1}'.format(processed, len(key_data))

            for i, feature_layer in enumerate(feature_layers):
                print 'converting ', feature_layer,' features to nd arrays...'
                features_np = np.array(features[i][0])
                features_np_norm = np.array(features[i][1])
                print feature_layer, ' features shape:'
                print 'writing ', feature_layer
                dump_results(model_folder, model_id, key, sub_key, feature_layer, features_np)
                dump_results(model_folder, model_id, key, sub_key, feature_layer + '_norm', features_np_norm)
    print 'done'

if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    txt_path = caffe_root + '/data/domain_adaptation_data/images/'
    save_path = caffe_root + '/data/domain_adaptation_data/results/'
    root_model_path = caffe_root + '/data/domain_adaptation_data/models/'
    mean_binary_path = caffe_root + '../data/models/alexnet/pretrained/places205CNN_mean.binaryproto'
    model_folder = 'freiburg_only'
    model_folder_path = root_model_path + model_folder + '/'
    deploy_path = model_folder_path + 'deploy.prototxt'
    caffe_model_path = model_folder_path + 'snapshots_iter_5000.caffemodel'
    batch_size = 1024
    input_layers = 'data_1'
    feature_layers = ['conv3', 'fc8_n']
    main()
