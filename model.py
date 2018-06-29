import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Lambda
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from settings import INPUT_SHAPE
from settings import TRAINING_DIR

from utils import batch_generator
from utils import load_data


def load_training_data():
    path = os.path.join(TRAINING_DIR, 'driving_log.csv')
    X, y = load_data(path)

    convert_path = lambda x: os.path.join(TRAINING_DIR, x.strip())
    X = [[convert_path(img) for img in row] for row in X]

    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.1)


def train(X, y, steps_per_epoch, epochs, batch_size, learning_rate):
    X_train, X_valid, y_train, y_valid = split_data(X, y)

    model = Sequential()

    f_normalize = lambda x: x/127.5 - 1.0
    model.add(Lambda(f_normalize, input_shape=INPUT_SHAPE))

    model.add(Conv2D(filters=24, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))

    model.add(Flatten())

    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
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

    model.fit_generator(
        generator=batch_generator(X_train, y_train, batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=batch_generator(X_valid, y_valid, batch_size),
        validation_steps=len(X_valid) / batch_size,
        callbacks=[checkpoint],
        verbose=1,
    )


def main():
    X, y = load_training_data()
    train(X, y, 20000, 10, 40, 1.0e-4)


if __name__ == '__main__':
    main()
