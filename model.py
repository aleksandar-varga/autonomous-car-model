import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Lambda
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from settings import IMAGE_SIZE
from setting import TRAINING_DIR

from utils import batch_generator
from utils import load_data


class NvidiaModel(Sequential):

    def __init__(self):
        super(NvidiaModel, self).__init__()

        # normalize images encoded using YUV color space
        f_normalize = lambda x: x/127.5 - 1.0
        self.add(Lambda(f_normalize, output_shape=IMAGE_SIZE))

        # subsample == stride
        self.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
        self.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
        self.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
        self.add(Conv2D(64, 3, 3, activation='relu'))
        self.add(Conv2D(64, 3, 3, activation='relu'))

        self.add(Flatten())

        self.add(Dense(100, activation='relu'))
        self.add(Dense(50, activation='relu'))
        self.add(Dense(10, activation='relu'))
        self.add(Dense(1))

        self.summary()


def load_training_data():
    path = os.path.join(TRAINING_DIR, 'driving_log.csv')
    X, y = load_data(path)

    convert_path = lambda x: os.path.join(TRAINING_DIR, x.strip())
    X = [[convert_path(img) for img in row] for row in X]

    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.1)


def train(X, y, steps_per_epoche, epoches, batch_size, learning_rate):
    X_train, y_train, X_valid, y_valid = split_data(X, y)

    checkpoint = ModelCheckpoint(
        'model-{epoch:03d}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto',
    )

    model = NvidiaModel()

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate),
    )

    model.fit_generator(
        generator=batch_generator(X_train, y_train, batch_size),
        steps_per_epoche=steps_per_epoche,
        epoches=epoches,
        validation_data=batch_generator(X_valid, y_valid, batch_size),
        validation_steps=len(X_valid) / batch_size,
        callbacks=[checkpoint],
        verbose=1,
    )


def main():
    X, y = load_training_data()
    X, y = X[20], y[20]
    train(X, y, 20000, 10, 40, 1.0e-4)


if __name__ == '__main__':
    main()
