import cv2
import numpy as np
import os
import pandas as pd
import random

from settings import IMAGE_CHANNELS
from settings import IMAGE_HEIGHT
from settings import IMAGE_WIDTH
from settings import TRAINING_DIR


def load_data(path):
    data = pd.read_csv(path, header=None)

    X = data[data.columns[:3]].values
    y = data[data.columns[3]].values

    return X, y


def load_image(path):
    img = cv2.imread(path)
    # convert to YUV color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # crop sky and car front
    img = img[60:-25]
    # resize to 200x66
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    return img


def batch_generator(X, y, batch_size):
    batch = (
        np.empty((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)),
        np.empty(batch_size)
    )
    pool = list(zip(X, y))
    while True:
        chosen = random.sample(pool, batch_size)
        for i in range(batch_size):
            # select either center, left or right image
            img_path = np.random.choice(chosen[i][0])
            batch[0][i] = load_image(img_path)
            batch[1][i] = chosen[i][1]
        yield batch


def main():
    data_path = os.path.join(TRAINING_DIR, 'driving_log.csv')
    X, y = load_data(data_path)

    convert_path = lambda x: os.path.join(TRAINING_DIR, x.strip())
    X = [[convert_path(img) for img in row] for row in X]

    g = batch_generator(X, y, 3)
    for _ in range(5):
        print(next(g))

if __name__ == '__main__':
    main()