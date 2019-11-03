"""
Assisted Listening Device
@Authors Jack Wang Reece Lehmann Akarsh Kumar

At HackTx 2019
"""


import os
os.environ["PATH"] += os.pathsep + "~/PycharmProjects/ALD/venv/lib/python3.6/site-packages/GraphViz"
import matplotlib as plt
import pickle as cPickle
import numpy as np
import librosa

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
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

        self.filters = 16
        self.kernel_size = (3, 3)
        self.padding = 'same'

        self.frequency_bins = 513
        self.frames = 100

        self.train_x = noisy_data_train
        self.train_y = clean_data_train

        self.test_x = noisy_data_test
        self.test_y = clean_data_test

        self.model = constructModel()

    def constructModel(self):
        model = Sequential()

        model.add(Conv2D(self.filters, self.kernel_size, padding=self.padding,
                         input_shape=(self.frequency_bins, self.frames, 1)))
        model.add(LeakyReLU())
        model.add(Conv2D(self.filters, self.kernel_size, padding=self.padding))
        model.add(LeakyReLU())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(.25))

        model.add(Conv2D(64, self.kernel_size, padding=self.padding))
        model.add(LeakyReLU())
        model.add(Conv2D(16, self.kernel_size, padding=self.padding))
        model.add(LeakyReLU())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU())
        model.add(Dropout(.25))
        model.add(Dense(513))

        return model
    def train(self):

        sgd = SGD(self.learning_rate, decay=1e-6, momentum=.9, nesterov=true)
        self.model.compile(optimizer=sgd,
                      loss='binary_crossentropy',
                      metric=['accuracy'])

        history = self.model.fit(self.train_x,
                            self.train_y,
                            batch_size=self.batch_size,
                            epochs=self.training_epochs)

        return history

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
    ALD1 = ALD(1, 1)
    # model1 = ALD1.model()
    # plot_model(model1, to_file='model.png')

    # SVG(model_to_dot(model1).create(prog='dot', format='svg'))