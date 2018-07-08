import argparse

import numpy
from keras.models import load_model

import utils
from settings import TEST_DIR


def calculateSSE(exp_results, giv_results):

    if len(exp_results) != len(giv_results):
        return False

    length = len(exp_results)

    sum = 0
    for i in range(length):
        difference = exp_results[i] - giv_results[i]
        sum +=  pow(difference,2)

    return sum


def calculateSST(expected_results, average):

    sum = 0
    for i in range(len(expected_results)):
        difference = average - expected_results[i]
        sum += pow(difference,2)

    return sum

def calculateSSR(giv_results, average):

    sum = 0
    for i in range(len(giv_results)):
        difference = average - giv_results[i]
        sum +=  pow(difference,2)

    return sum


def load_exptexted_results(path, given_results):
    model = load_model(path)
    return_list = []

    for path in given_results:
        image = utils.load_image(path)
        image = numpy.array([image])
        result = float(model.predict(image, batch_size=1))
        return_list.append(result)

    return return_list


def load_given_results():
    return utils.load_data(TEST_DIR, 'driving_log.csv')


def main(path):
    X, y = load_given_results()
    expected_results = load_exptexted_results(path, X)
    avg = numpy.mean(y)

    sse = calculateSSE(expected_results, y)
    sst = calculateSST(y, avg)
    ssr = calculateSSR(expected_results, avg)

    coefficient_of_determination_squared = ssr / sst

    print("sse: " + str(sse))
    print("sst: " + str(sst))
    print("ssr: " + str(ssr))

    print('coef:' + str(coefficient_of_determination_squared))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model')

    args = parser.parse_args()

    main(path=args.model)
