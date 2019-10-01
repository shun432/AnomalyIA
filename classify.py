import numpy as np
from sklearn import svm
from sklearn.neighbors import NearestNeighbors

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
        モデルを定義する場所
'''

class OneClassSVM:

    clf = svm.OneClassSVM(nu=0.03, kernel='rbf', gamma='auto')

    @classmethod
    def dofit(self, data, label):
        self.clf.fit(data, label)

    @classmethod
    def dopredict(self, data):
        pred = self.clf.predict(data)
        return pred

class kNN:

    NN = NearestNeighbors(n_neighbors=10)

    @classmethod
    def dofit(self, data):
        self.NN.fit(data)

    @classmethod
    def dopredict(self, data):
        dist, _ = self.NN.kneighbors(data)
        dist = dist / np.max(dist)
        return dist


class LPF:

    @classmethod
    def low_pass(self, past, now = 0, alpha = 0.1):
        return alpha * now + (1 - alpha) * past

    @classmethod
    def model(self, data):
        lowpassed = []
        lowpassed.append(0)
        for i in range(len(data)-1):
            lowpassed.append(self.low_pass(lowpassed[i], data[i]))
        return lowpassed


class LSTMs:

    '''
    http://cedro3.com/ai/keras-lstm/
    '''

    reference_len = 400

    @classmethod
    def preprocessing(self, raw_data, label):
        # read data
        raw_data = raw_data / 13

        # Make input data
        x, y = [], []
        length = self.reference_len
        for i in range(len(raw_data) - length):
            x.append(raw_data[i : i+length])
            y.append(raw_data[i + length])
        data = np.array(x).reshape(len(x), length, 1)
        target = np.array(y).reshape(len(y), 1)

        #labelの数の帳尻合わせ。length個のデータを参照、length番目のデータのラベルを予想
        for j in range(length-1):
            label = np.delete(label, 0)
        label = np.delete(label, len(label)-1)
        label = np.array(label).reshape(len(label), 1)

        return data, target, label

    @classmethod
    def modeling(self):
        # Model building
        length_of_sequence = self.reference_len
        in_out_neurons = 1
        n_hidden = 300

        model = Sequential()
        model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
        model.add(Dense(in_out_neurons))
        model.add(Activation("linear"))
        optimizer = Adam(lr=0.001)
        model.compile(loss="mean_squared_error", optimizer=optimizer)

        return model

    @classmethod
    def dofit(self, data, target, model, label):
        # Learning
        history = model.fit(data, label, batch_size=128, epochs=100, validation_split=0.1)
        predicted = model.predict(data)
        return history, predicted

    @classmethod
    def predict_future(self, data, model):
        # Future prediction
        future_test = data[len(data) - 1]
        future_result = []
        time_length = self.reference_len

        for i in range(24):
            test_data = np.reshape(future_test, (1, time_length, 1))
            batch_predict = model.predict(test_data)
            future_test = np.delete(future_test, 0)
            future_test = np.append(future_test, batch_predict)
            future_result = np.append(future_result, batch_predict)

        return future_result


class AutoEncoder:

    reference_len = 150

    @classmethod
    def preprocessing(self, raw_data):
        # read data
        raw_data = raw_data / 13

        # Make input data
        x, y = [], []
        length = self.reference_len
        for i in range(len(raw_data) - length):
            x.append(raw_data[i : i+length])
            y.append(raw_data[i + length])
        data = np.array(x).reshape(len(x), length, 1)
        target = np.array(y).reshape(len(y), 1)

        return data, target

    @classmethod
    def modeling(self):
        model = Sequential()

        model.add(Dense(128, activation='relu', input_shape=(self.reference_len, )))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.reference_len, activation='sigmoid'))
        model.summary()
        model.compile(loss='mse', optimizer='adam')

        return model

    @classmethod
    def dofit(self, data, model):
        history = model.fit(data, data, batch_size=128, verbose=1, epochs=20, validation_split=0.2)
        predicted = model.predict(data)
        return history, predicted
