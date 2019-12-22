import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam


class NeuralNetwork:

    '''
        https://keras.io/ja/getting-started/sequential-model-guide/
    '''

    def __init__(self, data_dim, epochs=10):

        self.data_dimension = data_dim
        self.epochs = epochs

        self.model = None

        #datashapeを使ってモデルを作っておく
        self.modeling()

    def modeling(self):

        in_out_neurons = 1
        n_hidden = 32

        model = Sequential()
        model.add(Dense(n_hidden, input_dim=self.data_dimension, activation='relu'))
        model.add(Dense(n_hidden, activation='relu'))
        model.add(Dense(in_out_neurons))
        model.add(Activation("linear"))
        optimizer = Adam(lr=0.001)
        model.compile(loss="mean_squared_error", optimizer=optimizer)

        self.model = model

    def preprocessing(self, data, target=None):

        # 最後からreference_steps個を保存
        data = np.array(data).reshape(1, self.data_dimension)

        # 最後からreference_steps個を保存
        if target is not None:
            target = np.array(target).reshape(1)

        return data, target

    def dofit(self, data, trend, epochs=None):
        if epochs is None:
            epochs = self.epochs
        # Learning
        history = self.model.fit(data, trend, epochs=epochs, verbose=0)
        return history

    def dopredict(self, data):
        predicted = self.model.predict(data)
        return predicted

