import shutil
import os
import glob
import ntpath


def is_file(path):
    return os.path.isfile(path)


def path_exists(path):
    return os.path.exists(path)


def copy(source_path, target_path):
    shutil.copy(source_path, target_path)


def rm_dir(path):
    shutil.rmtree(path)


def is_dir(path):
    return os.path.isdir(path)


def make_dir(path):
    os.makedirs(path)


def list_dir(path):
    return os.listdir(path)


def get_folder_contents(path, wild_card='*'):
    return glob.glob(path + wild_card)


def extract_name_from_path(path):
    return ntpath.basename(path)


def get_env_var(var, exception=True):
    if exception:
        return os.environ[var]
    else:
        return os.getenv(var)


def split_file_extension(var):
    return ntpath.splitext(var)


def path_rel_to_abs(rel_path, curr_dir_file ='.'):
    return os.path.abspath(os.path.join(os.path.dirname(curr_dir_file), rel_path))
