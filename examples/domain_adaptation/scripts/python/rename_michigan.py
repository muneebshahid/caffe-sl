import os_helper as osh


parser = argparse.ArgumentParser()
parser.add_argument("folder_path", help="Path to the michigan train folder")
args = parser.parse_args()

path_aug = args.folder_path + '/aug'
path_jan = args.folder_path + '/jan'

