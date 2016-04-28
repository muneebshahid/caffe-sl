import os_helper as osh


def convert(path, stride):
    new_path = path.replace('.txt', '.tri')
    with open(path, 'r') as triplet_file:
        lines = triplet_file.readlines()
        with open(new_path, 'w') as w:
            w.writelines([ lines[i][:-2] + ' ' + lines[i + 1][:-2] + ' ' + lines[i + 2][:-2] + '\n' for i in xrange(0, len(lines), 3)])
    return

def main():
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    image_root = caffe_root + '/data/domain_adaptation_data/images/'
    convert(image_root + 'triplet_data_train.txt', 3)
    convert(image_root + 'triplet_data_test.txt', 3)

if __name__ == "__main__":
    main()