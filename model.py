import argparse
import os
import json

import tensorflow as tf

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

from utils import load_image
from utils import load_data


def load_training_data():
    X, y = load_data(TRAINING_DIR, 'driving_log.csv')
    X = [load_image(path) for path in X]
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.1)


def train(X, y, steps_per_epoch, epochs, batch_size, learning_rate):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

    generator_options = dict(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )

    training_generator = ImageDataGenerator(**generator_options)
    training_generator.fit(X_train)

    validation_generator = ImageDataGenerator(**generator_options)
    validation_generator.fit(X_valid)

    model = Sequential()

    f_normalize = lambda x: x/127.5 - 1.0
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

    model.add(Dropout(0.33))

    model.add(Flatten())

    model.add(Dense(units=100, activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(units=50, activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(units=10, activation='linear'))
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
        generator=training_generator.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator.flow(X_valid, y_valid, batch_size=batch_size),
        validation_steps=len(X_valid) / batch_size,
        callbacks=[checkpoint],
        verbose=1,
    )

    with open('history.json', 'w') as f:
        json.dump(history, f, indent=1)


def main(gpu=False):
    X, y = load_training_data()

    with tf.device('/gpu:0' if gpu else '/cpu:0'):
        train(X, y, 10000, 8, 40, 1.0e-4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', default=False, action="store_true")

    args = parser.parse_args()

    main(args.gpu)
