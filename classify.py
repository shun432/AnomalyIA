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
from itertools import islice
plt.style.use('seaborn-whitegrid')


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
        history = model.fit(data, label, batch_size=128, epochs=30, validation_split=0.1)
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

    reference_len = 64

    offset = 13

    @classmethod
    def preprocessing(self, raw_data):
        # read data
        raw_data = (raw_data + self.offset/2) / self.offset

        # Make input data
        x, y = [], []
        length = self.reference_len
        for i in range(len(raw_data) - length):
            x.append(raw_data[i : i+length])
            y.append(raw_data[i + length])
        data = np.array(x).reshape(len(x), length)
        target = np.array(y).reshape(len(y))

        return data, target

    @classmethod
    def modeling(self):
        model = Sequential()

        model.add(Dense(round(self.reference_len/2), activation='relu', input_shape=(self.reference_len, )))
        model.add(Dense(round(self.reference_len/4), activation='relu'))
        model.add(Dense(round(self.reference_len/8), activation='relu'))
        model.add(Dense(round(self.reference_len/4), activation='relu'))
        model.add(Dense(round(self.reference_len/2), activation='relu'))
        model.add(Dense(round(self.reference_len), activation='sigmoid'))
        model.summary()
        model.compile(loss='mse', optimizer='adam')

        return model

    @classmethod
    def dofit(self, data, model):
        history = model.fit(data, data, batch_size=128, verbose=1, epochs=100, validation_split=0.2)
        predicted = model.predict(data)
        return history, predicted


class SSA:

    '''
    https://qiita.com/s_katagiri/items/d46448018fe2058d47da
    '''


    # SSA 用の関数
    @classmethod
    def window(self, seq, n):
        """
        window 関数で要素を1づつずらした2次元配列を出す. 戻り値は generator
        """
        "Returns a sliding window (of width n) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    @classmethod
    def SSA_anom(self, test, traject, w, ncol_t, ncol_h, ns_t, ns_h,
                 normalize=False):
        """
        特異スペクトル分析 (SSA) による時系列の特徴づけ
        ARGUMENTS:
        -------------------------------------------------
        test: array-like. テスト行列を作る部分時系列
        tracject: array-like. 履歴行列を作る部分時系列
        ns_h: 履歴行列から取り出す特異ベクトルの数
        ns_t: テスト行列から取り出す特異ベクトルの数
        -------------------------------------------------
        RETURNS:
        3要素のタプル:
            要素1: 2つの部分時系列を比較して求めた異常度
            要素2, 3: テスト行列・履歴行列をそれぞれの特異値の累積寄与率
        """
        H_test = np.array(
            tuple(x[:ncol_t] for x in self.window(test, w))[:w]
        )  # test matrix
        H_hist = np.array(
            tuple(x[:ncol_h] for x in self.window(traject, w))[:w]
        )  # trajectory matrix
        if normalize:
            H_test = (H_test - H_test.mean(axis=0,
                                           keepdims=True)) / H_test.std(axis=0)
            H_hist = (H_hist - H_hist.mean(axis=0,
                                           keepdims=True)) / H_hist.std(axis=0)
        Q, s1 = np.linalg.svd(H_test)[0:2]
        Q = Q[:, 0:ns_t]
        ratio_t = sum(s1[0:ns_t]) / sum(s1)
        U, s2 = np.linalg.svd(H_hist)[0:2]
        U = U[:, 0:ns_h]
        ratio_h = sum(s2[0:ns_t]) / sum(s2)
        anom = 1 - np.linalg.svd(np.matmul(U.T, Q),
                                 compute_uv=False
                                 )[0]
        return (anom, ratio_t, ratio_h)

    @classmethod
    def SSA_CD(self, series, w, lag,
               ncol_h=None, ncol_t=None,
               ns_h=None, ns_t=None,
               standardize=False, normalize=False, fill=True):
        """
        Change Detection by Singular Spectrum Analysis
        SSA を使った変化点検知
        -------------------------------------------------
        w   : window width (= row width of matrices) 短いほうが感度高くなる
        lag : default=round(w / 4)  Lag among 2 matrices 長いほうが感度高くなる
        ncol_h: 履歴行列の列数
        ncol_t: テスト行列の列数
        ns_h: 履歴行列から取り出す特異ベクトルの数. default=1 少ないほうが感度高くなる
        ns_t: テスト行列から取り出す特異ベクトルの数. default=1 少ないほうが感度高くなる
        standardize: 変換後の異常度の時系列を積分面積1で規格化するか
        fill: 戻り値の要素数を NaN 埋めで series と揃えるかどうか
        -------------------------------------------------
        Returns
        list: 3要素のリスト
            要素1: 2つの部分時系列を比較して求めた異常度のリスト
            要素2, 3: テスト行列・履歴行列をそれぞれの特異値の累積寄与率のリスト
        """
        if ncol_h is None:
            ncol_h = round(w / 2)
        if ncol_t is None:
            ncol_t = round(w / 2)
        if ns_h is None:
            ns_h = np.min([1, ncol_h])
        if ns_t is None:
            ns_t = np.min([1, ncol_t])
        if min(ncol_h, ncol_t) > w:
            print('ncol and ncol must be <= w')
        if ns_h > ncol_h or ns_t > ncol_t:
            print('I recommend to set ns_h >= ncol_h and ns_t >= ncol_t')
        start_at = lag + w + ncol_h
        end_at = len(series) + 1
        res = []
        for t in range(start_at, end_at):
            res = res + [self.SSA_anom(series[t - w - ncol_t:t],
                                  series[t - lag - w - ncol_h:t - lag],
                                  w=w, ncol_t=ncol_t, ncol_h=ncol_h,
                                  ns_t=ns_t, ns_h=ns_h,
                                  normalize=normalize)]
        anom = [round(x, 14) for x, r1, r2 in res]
        ratio_t = [r1 for x, r1, r2 in res]
        ratio_h = [r2 for x, r1, r2 in res]
        if fill == True:
            anom = [np.nan] * (start_at - 1) + anom
        if standardize:
            c = np.nansum(anom)
            if c != 0:
                anom = [x / c for x in anom]
        return [anom, ratio_t, ratio_h]
