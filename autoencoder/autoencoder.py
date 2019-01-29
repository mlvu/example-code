import keras

from keras.optimizers import Adam
from tensorflow.python.client import device_lib

from sklearn import datasets
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from keras.layers import Dense, Reshape, Flatten
from keras.models import Sequential

# This function contains all the code
def go(options):

    # Debugging info to see if we're using the GPU
    print('devices', device_lib.list_local_devices())

    # These are people in the data that smile
    SMILING = [0, 7, 8, 11, 12, 13, 14, 20, 27, 155, 153, 154, 297]
    NONSMILING = [1, 2, 3, 6, 10, 60, 61, 136, 138, 216, 219, 280]

    # Dowload the data
    faces = datasets.fetch_lfw_people(data_home='.')
    x = faces.images # x is a 13000 by 67 by 42 array

    hidden_size = options.hidden

    # Build the encoder
    encoder = Sequential()

    encoder.add(Flatten(input_shape=(62, 47)))
    encoder.add(Dense(1024, activation='relu'))
    encoder.add(Dense(512, activation='relu'))
    encoder.add(Dense(256, activation='relu'))
    encoder.add(Dense(128, activation='relu'))
    encoder.add(Dense(hidden_size))

    # Build the decoder
    decoder = Sequential()

    decoder.add(Dense(128, activation='relu', input_dim=hidden_size))
    decoder.add(Dense(256, activation='relu'))
    decoder.add(Dense(512, activation='relu'))
    decoder.add(Dense(1024, activation='relu'))
    decoder.add(Dense(62*47, activation='relu'))
    decoder.add(Reshape((62,47)))

    # Stick em together to make the autoencoder
    auto = Sequential()

    auto.add(encoder)
    auto.add(decoder)

    auto.summary()

    # Choose a loss function (MSE) and a search algorithm
    #         (Adam, a fancy version of gradient descent)
    optimizer = Adam(lr=options.lr)
    auto.compile(optimizer=optimizer, loss='mse')

    # Search for a good model
    auto.fit(x, x,
             epochs=options.epochs,
             batch_size=256)

    # Select the smiling and nonsmiling images from the dataset
    smiling = x[SMILING, ...]
    nonsmiling = x[NONSMILING, ...]

    # Pass them through the encoder
    smiling_latent = encoder.predict(smiling)
    nonsmiling_latent = encoder.predict(nonsmiling)

    # Compute the means for both groups
    smiling_mean = smiling_latent.mean(axis=0)
    nonsmiling_mean = nonsmiling_latent.mean(axis=0)

    # Subtract for smiling vector
    smiling_vector = smiling_mean - nonsmiling_mean


    # Making somebody smile (person 42):
    latent = encoder.predict(x[None, 42, ...])
    l_smile  = latent + 0.3 * smiling_vector
    smiling = decoder.predict(l_smile)

    # Plot fronwing-to-smiling transition for several people
    # in a big PDF image
    randos = 6
    k = 9
    fig = plt.figure(figsize=(k, randos))

    for rando in range(randos):
        rando_latent = encoder.predict(x[None, rando, ...])

        # plot several images
        adds = np.linspace(-1.0, 1.0, k)

        for i in range(k):
            gen_latent = rando_latent + adds[i] * smiling_vector
            gen = decoder.predict(gen_latent)

            ax = fig.add_subplot(randos, k, rando * k + i + 1, xticks=[], yticks=[])
            ax.imshow(gen.reshape((62, 47)), cmap=plt.cm.gray)

    plt.savefig('rando-to-smiling.pdf')


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.01, type=float)

    parser.add_argument("-H", "--hidden-size",
                        dest="hidden",
                        help="Latent vector size",
                        default=64, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)
