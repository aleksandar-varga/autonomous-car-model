import os
import random

import cv2
import numpy as np
import pandas as pd

from settings import IMAGE_CHANNELS
from settings import IMAGE_HEIGHT
from settings import IMAGE_WIDTH
from settings import TRAINING_DIR


def load_data(dir, file):
    path = os.path.join(dir, file)
    data = pd.read_csv(path, header=None)

    X = data[data.columns[:3]].values.flatten().tolist()
    y = []
    for record in data[data.columns[3]].values:
        y.extend([record, record + 0.25, record - 0.25])
   
    convert_path = lambda x: os.path.join(dir, x.strip())
    X = [convert_path(img) for img in X]

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
            batch[0][i] = load_image(chosen[i][0])
            batch[1][i] = chosen[i][1]
        yield batch
