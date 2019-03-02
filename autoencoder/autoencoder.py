import keras, sys

from keras.optimizers import Adam
from tensorflow.python.client import device_lib

from sklearn import datasets
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from keras import backend as K
from keras import layers as kl
from keras.models import Sequential, Model

from keras.preprocessing.image import ImageDataGenerator

from tqdm import trange, tqdm

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def prod(inp):
    res = 1
    for i in inp:
        res *= i
    return res

def gather(generator, batches):
    res = []
    for i, batch in enumerate(generator):
        res.append(batch[0])
        if i > batches:
            break

    return np.concatenate(res, axis=0)


def l1_loss(y_true, y_pred):
    losses = K.abs(y_true - y_pred)
    return K.mean(K.sum(K.sum(K.sum(losses, axis=3), axis=2), axis=1), axis=0) # Note the sum over pixels and channels, so the loss stays balanced with the kl term

def l2_loss(y_true, y_pred):
    losses = K.square(y_true - y_pred)
    return K.mean(K.sum(K.sum(K.sum(losses, axis=3), axis=2), axis=1), axis=0) # Note the sum over pixels and channels, so the loss stays balanced with the kl term

def bce_loss(y_true, y_pred):
    losses = K.binary_crossentropy(y_true, y_pred)
    return K.mean(K.sum(K.sum(K.sum(losses, axis=3), axis=2), axis=1), axis=0) # Note the sum over pixels and channels, so the loss stays balanced with the kl term


class KLLayer(kl.Layer):

    """
    Identity transform layer that adds KL divergence
    to the final model loss.
    During training, call
            K.set_value(kl_layer.weight, new_value)
    to scale the KL loss term.
    based on:
    http://tiao.io/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
    """

    def __init__(self, weight = None, *args, **kwargs):
        self.is_placeholder = True
        self.weight = weight
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        loss =  K.mean(kl_batch)
        if self.weight is not None:
            loss = loss * self.weight

        self.add_loss(loss, inputs=inputs)

        return inputs

class Sample(kl.Layer):
    """
    Performs VAE sampling step
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var, eps = inputs

        z = K.exp(.5 * log_var) * eps + mu

        return z

    def compute_output_shape(self, input_shape):
        shape_mu, _, _ = input_shape

        return shape_mu

def go(options):

    # Debugging info to see if we're using the GPU
    print('devices', device_lib.list_local_devices())

    if options.loss == 'bce':
        loss = bce_loss
    elif options.loss == 'l1':
        loss = l1_loss
    elif options.loss == 'l2':
        loss = l2_loss
    else:
        raise Exception('Loss type {} not recognized.'.format(options.loss))

    if options.dataset == 'lfw':

        datagen = ImageDataGenerator(rescale=1./255)

        # These are people in the data that smile
        SMILING = [0, 7, 8, 11, 12, 13, 14, 20, 27, 155, 153, 154, 297]
        NONSMILING = [1, 2, 3, 6, 10, 60, 61, 136, 138, 216, 219, 280]

        # Dowload the data
        faces = datasets.fetch_lfw_people(data_home=options.data_dir)
        x = np.pad(faces.images, ((0,0), (1,1), (0, 1)), mode='constant')[:, :, :, None]

        shape = x.shape[1:]
        xgen = datagen.flow(x, y=x, batch_size=options.batch_size, shuffle=False)
        pooling = 2
        grayscale = True

        size = x.shape[0]
        print('Using LFW dataset, {} instances.'.format(x.shape[0]))

    elif options.dataset == 'ffhq':
        datagen = ImageDataGenerator(
            rescale=1. / 255)

        # These are people in the data that smile
        SMILING = [1, 6, 7, 9, 11, 14, 17, 18, 19, 22, 25, 30, 32, 37, 38, 45, 47, 49, 55, 56]
        NONSMILING = [2, 4, 12, 20, 21, 24, 26, 41, 43, 44, 48, 51, 52, 53, 58, 59, 63, 68, 69, 98]

        # Dowload the data
        xgen = datagen.flow_from_directory(options.data_dir, batch_size=options.batch_size, target_size=(128, 128), shuffle=False)
        shape = (128, 128, 3)
        pooling = 4
        grayscale = False

        print('Using FFHQ thumbs dataset, {} batches.'.format(len(xgen)))

    ##-- Plot the data

    # extract the first 500 faces into a tensor
    faces = gather(xgen, 500 // options.batch_size)

    # plot data
    fig = plt.figure(figsize=(5, 20))
    for i in range(5 * 20):
        ax = fig.add_subplot(20, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(faces[i] * (np.ones(3) if grayscale else 1))
        ax.set_title(i)

    plt.tight_layout()
    plt.savefig('faces.pdf')

    # smiling/nonsmiling
    fig = plt.figure(figsize=(5, 4))
    for i in range(len(SMILING)):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(faces[SMILING[i]] * (np.ones(3) if grayscale else 1))

    plt.savefig('smiling-faces.pdf')

    fig = plt.figure(figsize=(5, 4))
    for i in range(len(NONSMILING)):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(faces[NONSMILING[i]] * (np.ones(3) if grayscale else 1))

    plt.savefig('nonsmiling-faces.pdf')

    ##-- Build the model
    hidden_size = options.hidden

    # Build the encoder
    encoder = Sequential()

    a, b, c = 16, 32, 128
    vmult = 2 if options.variational else 1

    encoder.add(kl.Conv2D(kernel_size=(3, 3), filters=a, padding='same', activation='relu', input_shape=shape))
    encoder.add(kl.Conv2D(kernel_size=(3, 3), filters=a, padding='same', activation='relu'))
    encoder.add(kl.Conv2D(kernel_size=(3, 3), filters=a, padding='same', activation='relu'))
    encoder.add(kl.MaxPool2D(pool_size=(pooling, pooling)))
    encoder.add(kl.Conv2D(kernel_size=(3, 3), filters=b, padding='same', activation='relu'))
    encoder.add(kl.Conv2D(kernel_size=(3, 3), filters=b, padding='same', activation='relu'))
    encoder.add(kl.Conv2D(kernel_size=(3, 3), filters=b, padding='same', activation='relu'))
    encoder.add(kl.MaxPool2D(pool_size=(pooling, pooling)))
    encoder.add(kl.Conv2D(kernel_size=(3, 3), filters=c, padding='same', activation='relu'))
    encoder.add(kl.Conv2D(kernel_size=(3, 3), filters=c, padding='same', activation='relu'))
    encoder.add(kl.Conv2D(kernel_size=(3, 3), filters=c, padding='same', activation='relu'))
    encoder.add(kl.MaxPool2D(pool_size=(pooling, pooling)))
    encoder.add(kl.Flatten())
    encoder.add(kl.Dense(hidden_size * 5, activation='relu'))
    encoder.add(kl.Dense(hidden_size * vmult))

    encoder.summary()

    # Build the decoder
    lower_shape = (shape[0] // (pooling ** 3), shape[1] // (pooling ** 3), c)
    print(lower_shape, prod(lower_shape))
    decoder = Sequential()
    decoder.add(kl.Dense(hidden_size * 5, activation='relu', input_dim=hidden_size))
    decoder.add(kl.Dense(prod(lower_shape), activation='relu'))
    decoder.add(kl.Reshape(lower_shape))
    decoder.add(kl.UpSampling2D(size=(pooling, pooling)))
    decoder.add(kl.Conv2DTranspose(kernel_size=(3, 3), filters=c, padding='same', activation='relu'))
    decoder.add(kl.Conv2DTranspose(kernel_size=(3, 3), filters=c, padding='same', activation='relu'))
    decoder.add(kl.Conv2DTranspose(kernel_size=(3, 3), filters=c, padding='same', activation='relu'))
    decoder.add(kl.UpSampling2D(size=(pooling, pooling)))
    decoder.add(kl.Conv2DTranspose(kernel_size=(3, 3), filters=b, padding='same', activation='relu'))
    decoder.add(kl.Conv2DTranspose(kernel_size=(3, 3), filters=b, padding='same', activation='relu'))
    decoder.add(kl.Conv2DTranspose(kernel_size=(3, 3), filters=b, padding='same', activation='relu'))
    decoder.add(kl.UpSampling2D(size=(pooling, pooling)))
    decoder.add(kl.Conv2DTranspose(kernel_size=(3, 3), filters=a, padding='same', activation='relu'))
    decoder.add(kl.Conv2DTranspose(kernel_size=(3, 3), filters=a, padding='same', activation='relu'))
    decoder.add(kl.Conv2DTranspose(kernel_size=(3, 3), filters=a, padding='same', activation='relu'))
    decoder.add(kl.Conv2D(kernel_size=(1, 1), filters=shape[2], padding='same', activation='sigmoid'))

    decoder.summary()

    if not options.variational:
        # Stick em together to make the autoencoder
        auto = Sequential()

        auto.add(encoder)
        auto.add(decoder)
    else:
        auto = Model()

        xin = kl.Input(shape=shape)
        eps = kl.Input(tensor=K.random_normal(shape=(K.shape(xin)[0], hidden_size)))

        z = encoder(xin)

        # slice out the zmean and zvar
        zmean = kl.Lambda(lambda x: x[:, :hidden_size], output_shape=(hidden_size,))(z)
        zvar  = kl.Lambda(lambda x: x[:, hidden_size:], output_shape=(hidden_size,))(z)

        zmean, zvar = KLLayer()([zmean, zvar])

        zsample = Sample()([zmean, zvar, eps])

        out = decoder(zsample)

        auto = Model([xin, eps], out)

    # Choose a loss function (BCE) and a search algorithm
    optimizer = Adam(lr=options.lr)
    auto.compile(optimizer=optimizer, loss=loss)

    ##-- Training
    plotat = [0, 2, 5, 10, 25, 50, 75, 100, 150, 250] # plot reconstructions for these epochs
    for e in range(options.epochs):
        print('EPOCH ', e)
        if e in plotat:
            # plot reconstructions
            rec = auto.predict(faces[:5*20])

            print('faces', faces[0, :5, :5, 0])
            print('rec', rec[0, :5, :5, 0])

            fig = plt.figure(figsize=(5, 20))
            for i in range(5 * 20):
                ax = fig.add_subplot(20, 5, i + 1, xticks=[], yticks=[])
                ax.imshow(rec[i] * (np.ones(3) if grayscale else 1))

            plt.tight_layout()
            plt.savefig('reconstructions.{:04}.pdf'.format(e))
        for i, batch in tqdm(enumerate(xgen)):
            auto.train_on_batch(batch[0], batch[0])
            if i > len(xgen):
                break

    # Select the smiling and nonsmiling images from the dataset
    smiling = faces[SMILING, ...]
    nonsmiling = faces[NONSMILING, ...]

    # Pass them through the encoder
    smiling_latent = encoder.predict(smiling)
    nonsmiling_latent = encoder.predict(nonsmiling)

    # Compute the means for both groups
    smiling_mean = smiling_latent.mean(axis=0)
    nonsmiling_mean = nonsmiling_latent.mean(axis=0)

    # Subtract for smiling vector
    smiling_vector = smiling_mean - nonsmiling_mean

    # # Making somebody smile (person 42):
    # latent = encoder.predict(faces[None, 42, ...])
    # l_smile  = latent + 0.3 * smiling_vector
    # smiling = decoder.predict(l_smile)

    # Plot frowning-to-smiling transition for several people
    # in a big PDF image
    randos = 6
    k = 9
    fig = plt.figure(figsize=(k, randos))

    for rando in range(randos):
        rando_latent = encoder.predict(faces[rando:rando+1])

        # plot several images
        adds = np.linspace(-1.0, 1.0, k)

        for i in range(k):
            gen_latent = rando_latent + adds[i] * smiling_vector
            gen = decoder.predict(gen_latent)

            ax = fig.add_subplot(randos, k, rando * k + i + 1, xticks=[], yticks=[])
            ax.imshow(gen[0] * (np.ones(3) if grayscale else 1), cmap=plt.cm.gray)

    plt.savefig('rando-to-smiling.pdf')


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("-b", "--batch",
                        dest="batch_size",
                        help="Batch size.",
                        default=32
                        , type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.005, type=float)

    parser.add_argument("-H", "--hidden-size",
                        dest="hidden",
                        help="Latent vector size",
                        default=128, type=int)

    parser.add_argument("-D", "--dataset",
                        dest="dataset",
                        help="Name of the dataset [lfw, ffhq]",
                        default='lfw', type=str)

    parser.add_argument("-d", "--data-dir",
                        dest="data_dir",
                        help="Data directory (for lfw the data will downloaded here, for ffhq, you should download the data and put it here)",
                        default='./data', type=str)

    parser.add_argument("-V", "--variational",
                        dest="variational",
                        help="Use a variational autoencoder",
                        action='store_true')

    parser.add_argument("-L", "--loss",
                        dest="loss",
                        help="Reconstruction loss to use (bce, l1, l2).",
                        default='bce', type=str)


    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)
