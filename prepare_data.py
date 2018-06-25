import argparse
import os
import shutil
import sys
import tempfile
import zipfile

from sklearn.model_selection import train_test_split

import settings


def main(path, force=False):
    if os.path.exists(settings.DATA_DIR):
        if force:
            print('Clearing data directory..')
            shutil.rmtree(settings.DATA_DIR)
        else:
            print('Data folder already exists. If you would like to clean ' + \
                'it up, use -f or --force argument.')
            exit(1)

    if not os.path.exists(path):
        raise Exception(path + ' does not exist.')

    if not os.path.splitext(path)[1] == '.zip':
        raise Exception(path + ' is not a zip file.')

    init_dirs()

    temp_dir = tempfile.mkdtemp()

    print('Extracting data..')
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(temp_dir)

    split_data(temp_dir)

    print('Cleaning up..')
    shutil.rmtree(temp_dir)

def init_dirs():
    print('Creating directories..')
    os.mkdir(settings.DATA_DIR)
    os.mkdir(settings.TRAINING_DIR)
    os.mkdir(os.path.join(settings.TRAINING_DIR, 'IMG'))
    os.mkdir(settings.TEST_DIR)
    os.mkdir(os.path.join(settings.TEST_DIR, 'IMG'))

def split_data(path):
    tmp_data_dir = os.path.join(path, 'data')
    file_path = os.path.join(tmp_data_dir, 'driving_log.csv')

    with open(file_path, 'r') as f:
        data = f.readlines()[1:]
        training, test = train_test_split(data, test_size=0.2)

        print('Moving data to training directory..')
        move_data(training, tmp_data_dir, settings.TRAINING_DIR)
        print('Moving data to test directory..')
        move_data(test, tmp_data_dir, settings.TEST_DIR)

def move_data(data, source, destination):
    file_path = os.path.join(destination, 'driving_log.csv')

    with open(file_path, 'w') as f:
        f.writelines(data)

    for point in data:
        for image_path in point.split(',')[:3]:
            src = os.path.join(source, image_path.strip())
            dst = os.path.join(destination, image_path.strip())

            shutil.move(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('path')
    parser.add_argument('-f', '--force', default=False, action="store_true")

    args = parser.parse_args()

    main(args.path, args.force)
