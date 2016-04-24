from data_loader import DataLoader


def get_train_test_split_len(examples_len, split):
    # put first split% of the data for training and the rest for testing
    return int(np.ceil(split * examples_len)), -int(np.floor((1 - split) * examples_len))


def split_train_test(data_set, split=0.7):
    train_examples, test_examples = get_train_test_split_len(len(data_set), split)
    # ensures we do not append the same sequence again
    return data_set[0:train_examples], data_set[train_examples:]

def process_michigan(dataset):
    processed_dataset = []
    for instance in dataset:
        processed_dataset.append(instance[0])
        processed_dataset.append(instance[1])
    return


def process_nordland(dataset):
    processed_dataset = []

def process_freiburg(dataset):
    processed_dataset = []
    last_summer = None
    last_winter = None
    for instance in dataset:
        summer, winter = instance[0], instance[1]
        if summer != last_summer:
            processed_dataset.append(summer)
        if winter != last_winter:
            processed_dataset.append(winter)
        last_summer, last_winter = summer, winter
    return processed_dataset


def main():
    dataset = []
    keys = ['freiburg', 'michigan', 'nordland', 'alderly', 'kitti']
    for key in keys:
        train_set, _ = split_train_test(DataLoader.load(key))
        if key == 'freiburg':
            dataset = process_freiburg(train_set)
        elif key == 'michigan'
            dataset = process_michigan(train_set)

if __name__ == '__main__':
    main()