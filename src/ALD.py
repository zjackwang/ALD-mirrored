"""
Assisted Listening Device
@Authors Jack Wang Reece Lehmann Akarsh Kumar

At HackTx 2019
"""


import os

from keras import Model, Input
from keras.layers.convolutional import Deconv2D

os.environ["PATH"] += os.pathsep + "~/PycharmProjects/ALD/venv/lib/python3.6/site-packages/GraphViz"
import matplotlib as plt
import pickle as cPickle
import numpy as np
import librosa

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Concatenate
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy
from keras.activations import softmax

from keras.utils import plot_model
from IPython.display import SVG
from keras.utils import model_to_dot

class ALD:
    def __init__(self, noisy_data_train, clean_data_train, noisy_data_test, clean_data_test):
        self.batch_size = 64
        self.training_epochs = 50
        self.learning_rate = .005
        self.alpha = 0.2

        self.padding = 'same'

        self.frequency_bins = 512
        self.frames = 128

        self.train_x = noisy_data_train
        self.train_y = clean_data_train

        self.test_x = noisy_data_test
        self.test_y = clean_data_test

        self.model = self.construct_model()

    def construct_model(self):

        inputs = Input(shape=(self.frequency_bins, self.frames, 1))

        conv1 = Conv2D(16, 5, strides=2, padding=self.padding)(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)

        conv2 = Conv2D(32, 5, strides=2, padding=self.padding)(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)

        conv3 = Conv2D(64, 5, strides=2, padding=self.padding)(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)

        conv4 = Conv2D(128, 5, strides=2, padding=self.padding)(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=0.2)(conv4)

        conv5 = Conv2D(256, 5, strides=2, padding=self.padding)(conv4)
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=0.2)(conv5)

        conv6 = Conv2D(512, 5, strides=2, padding=self.padding)(conv5)
        conv6 = BatchNormalization()(conv6)
        conv6 = LeakyReLU(alpha=0.2)(conv6)

        deconv7 = Deconv2D(256, 5, strides=2, padding=self.padding)(conv6)
        deconv7 = BatchNormalization()(deconv7)
        deconv7 = Dropout(0.5)(deconv7)
        deconv7 = Activation('relu')(deconv7)

        deconv8 = Concatenate(axis=3)([deconv7, conv5])
        deconv8 = Deconv2D(128, 5, strides=2, padding=self.padding)(deconv8)
        deconv8 = BatchNormalization()(deconv8)
        deconv8 = Dropout(0.5)(deconv8)
        deconv8 = Activation('relu')(deconv8)

        deconv9 = Concatenate(axis=3)([deconv8, conv4])
        deconv9 = Deconv2D(64, 5, strides=2, padding=self.padding)(deconv9)
        deconv9 = BatchNormalization()(deconv9)
        deconv9 = Dropout(0.5)(deconv9)
        deconv9 = Activation('relu')(deconv9)

        deconv10 = Concatenate(axis=3)([deconv9, conv3])
        deconv10 = Deconv2D(32, 5, strides=2, padding=self.padding)(deconv10)
        deconv10 = BatchNormalization()(deconv10)
        deconv10 = Activation('relu')(deconv10)

        deconv11 = Concatenate(axis=3)([deconv10, conv2])
        deconv11 = Deconv2D(16, 5, strides=2, padding=self.padding)(deconv11)
        deconv11 = BatchNormalization()(deconv11)
        deconv11 = Activation('relu')(deconv11)

        deconv12 = Concatenate(axis=3)([deconv11, conv1])
        deconv12 = Deconv2D(1, 5, strides=2, padding=self.padding)(deconv12)
        deconv12 = Activation('relu')(deconv12)

        model = Model(inputs, deconv12)

        return model

    def sample(self, num_dirty, num_clean):
        n = len(num_dirty)
        mini_batches = [
            [num_dirty[k:k+self.batch_size], num_clean[k:k+self.batch_size]]
            for k in range(0, n, self.batch_size)
        ]

        return mini_batches

    def train(self, model):
        sgd = SGD(self.learning_rate, decay=1e-6, momentum=.9, nesterov=True)
        model.compile(optimizer=sgd,
                      loss='binary_crossentropy')

        for epoch in range(self.training_epochs):
            mini_batches = self.sample(self.train_x, self.train_y)
            for batch in mini_batches:
                X, y = batch[0], batch[1]
                model.fit(X, y, batch_size=self.batch_size, verbose=True, validation_split=0.075)
                model.save('vocal_{:0>2d}.h5'.format(epoch+1), overwrite=True)



    def eval(self, model):
        score = model.evaluate(self.test_x,
                               self.test_y,
                               batch_size=self.batch_size)

    def stats(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

if __name__ == "__main__":
    # ALD1 = ALD(1, 1)
    # model1 = ALD1.model()
    # plot_model(model1, to_file='model.png')

    # SVG(model_to_dot(model1).create(prog='dot', format='svg'))