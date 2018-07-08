import argparse
import os
import json

import tensorflow as tf
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from settings import INPUT_SHAPE
from settings import TRAINING_DIR

from utils import batch_generator
from utils import load_image
from utils import load_data


def load_training_data():
    X, y = load_data(TRAINING_DIR, 'driving_log.csv')
    X = np.array([load_image(path) for path in X])
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.1)


def train(X, y, batch_size=32, epochs=10, steps_per_epoch=10000, learning_rate=0.001):
    X_train, X_valid, y_train, y_valid = split_data(X, y)

    model = Sequential()

    f_normalize = lambda x: x / 127.5 - 1.0
    model.add(Lambda(f_normalize, input_shape=INPUT_SHAPE))

    model.add(Conv2D(filters=24, kernel_size=5, strides=2, activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=36, kernel_size=5, strides=2, activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=48, kernel_size=5, strides=2, activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=3, activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=3, activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(units=120, activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=70, activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=12, activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(units=1))

    model.summary()

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate),
    )

    checkpoint = ModelCheckpoint(
        'model-{epoch:03d}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto',
    )

    history = model.fit_generator(
        generator=batch_generator(X_train, y_train, batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=batch_generator(X_valid, y_valid, batch_size, shuffle=False),
        validation_steps=len(X_valid) / batch_size,
        callbacks=[checkpoint],
        verbose=1,
    )

    with open('history.json', 'w') as f:
        json.dump(history.history, f, indent=4)


def main(gpu=False, batch_size=32, epochs=10, steps_per_epoch=10000, learning_rate=0.001):
    X, y = load_training_data()

    with tf.device('/gpu:0' if gpu else '/cpu:0'):
        train(X, y, batch_size, epochs, steps_per_epoch, learning_rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', default=False, action="store_true")

    parser.add_argument('-b', '--batch-size', default=32, action="store")
    parser.add_argument('-e', '--epochs', default=10, action="store")
    parser.add_argument('-s', '--steps-per-epoch', default=10000, action="store")
    parser.add_argument('-l', '--learning-rate', default=10, action="store")
    
    args = parser.parse_args()

    main(
        args.gpu, 
        float(args.batch_size), 
        int(args.epochs), 
        int(args.steps_per_epoch), 
        float(args.learning_rate)
    )
