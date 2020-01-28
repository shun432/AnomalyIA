import numpy as np
from sklearn import svm
from sklearn.neighbors import NearestNeighbors

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from itertools import islice
plt.style.use('seaborn-whitegrid')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


'''
        モデルを定義する場所
'''

class OneClassSVM:

    def __init__(self):
        self.clf = svm.OneClassSVM(nu=0.03, kernel='rbf', gamma='auto')

    def dofit(self, data, label):
        self.clf.fit(data, label)

    def dopredict(self, data):
        pred = self.clf.predict(data)
        return pred


class kNN:

    def __init__(self):
        self.NN = NearestNeighbors(n_neighbors=10)

    def dofit(self, data):
        self.NN.fit(data)

    def dopredict(self, data):
        dist, _ = self.NN.kneighbors(data)
        dist = dist / np.max(dist)
        return dist


class LPF:

    def low_pass(self, past, now = 0, alpha = 0.1):
        return alpha * now + (1 - alpha) * past

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

    def __init__(self):
        self.reference_len = 400
        self.offset = 13

    def preprocessing(self, raw_data, label):
        # read data
        raw_data = raw_data / self.offset

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

    def dofit(self, data, target, model, label):
        # Learning
        history = model.fit(data, label, batch_size=128, epochs=30, validation_split=0.1)
        predicted = model.predict(data)
        return history, predicted

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



class LSTMa:

    '''
    http://cedro3.com/ai/keras-lstm/
    '''

    def __init__(self, data_dim, timesteps=10, epochs=10):

        self.reference_steps = timesteps
        self.data_dimension = data_dim
        self.epochs = epochs

        self.model = None

        #datashapeを使ってモデルを作っておく
        self.modeling()

    # LSTMに入力するreference_steps個のデータだけを参照するようにdataとtargetを整える
    def preprocessing(self, data, target=None, reference_offset=1):

        # 最後からreference_steps個を保存
        data = np.array([u[- self.reference_steps - reference_offset: - reference_offset] for u in data]).T.reshape(self.reference_steps, self.data_dimension, 1)

        # 最後からreference_steps個を保存
        if target is not None:
            target = np.array(target[- self.reference_steps - reference_offset: - reference_offset]).reshape(self.reference_steps, 1)

        return data, target

    def modeling(self):
        # Model building
        in_out_neurons = 1
        n_hidden = 32

        model = Sequential()
        model.add(LSTM(n_hidden, return_sequences=True))
        model.add(LSTM(n_hidden, return_sequences=True))
        model.add(LSTM(n_hidden))
        model.add(Dense(in_out_neurons))
        model.add(Activation("linear"))
        optimizer = Adam(lr=0.001)
        model.compile(loss="mean_squared_error", optimizer=optimizer)

        self.model = model

    def dofit(self, data, trend, epochs=None):
        if epochs is None:
            epochs = self.epochs
        # Learning
        history = self.model.fit(data, trend, epochs=epochs, validation_split=0.1, verbose=0)
        return history

    def dopredict(self, data):
        predicted = self.model.predict(data)
        return predicted

    # def predict_future(self, data, model):
    #     # Future prediction
    #     future_test = data[len(data) - 1]
    #     future_result = []
    #     time_length = self.reference_len
    #
    #     for i in range(24):
    #         test_data = np.reshape(future_test, (1, time_length, 1))
    #         batch_predict = model.predict(test_data)
    #         future_test = np.delete(future_test, 0)
    #         future_test = np.append(future_test, batch_predict)
    #         future_result = np.append(future_result, batch_predict)
    #
    #     return future_result



class AutoEncoder:

    def __init__(self):
        self.reference_len = 64
        self.offset = 13

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

    def modeling(self):
        model = Sequential()

        model.add(Dense(round(self.reference_len/2), activation='relu', input_shape=(self.reference_len, )))
        model.add(Dense(round(self.reference_len/4), activation='relu'))
        # model.add(Dense(round(self.reference_len/8), activation='relu'))
        # model.add(Dense(round(self.reference_len/16), activation='relu'))
        model.add(Dense(round(self.reference_len/8), activation='relu'))
        model.add(Dense(round(self.reference_len/4), activation='relu'))
        model.add(Dense(round(self.reference_len/2), activation='relu'))
        model.add(Dense(round(self.reference_len), activation='sigmoid'))
        model.summary()
        model.compile(loss='mse', optimizer='adam')

        return model

    def dofit(self, data, model):
        history = model.fit(data, data, batch_size=128, verbose=1, epochs=100, validation_split=0.2)
        predicted = model.predict(data)
        return history, predicted


class SSA:

    '''
    https://qiita.com/s_katagiri/items/d46448018fe2058d47da
    '''


    # SSA 用の関数
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


class SVRs:

    '''
    https://qiita.com/koshian2/items/baa51826147c3d538652
    '''

    def preprocessing(self, X):
        # 訓練データを基準に標準化（平均、標準偏差で標準化）
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        # # テストデータも標準化
        # Xtest_norm = scaler.transform(Xtest)

        return X_norm

    # 6:2:2に分割にするため、訓練データのうちの後ろ1/4を交差検証データとする
    # 交差検証データのジェネレーター
    def gen_cv(self, y):
        m_train = np.floor(len(y) * 0.75).astype(int)  # このキャストをintにしないと後にハマる
        train_indices = np.arange(m_train)
        test_indices = np.arange(m_train, len(y))
        yield (train_indices, test_indices)

    def tuning(self, X_norm, y):
        # ハイパーパラメータのチューニング
        params_cnt = 20
        params = {"C": np.logspace(0, 2, params_cnt), "epsilon": np.logspace(-1, 1, params_cnt)}
        gridsearch = GridSearchCV(SVR(), params, cv=self.gen_cv(y), scoring="r2", return_train_score=True)
        gridsearch.fit(X_norm, y)
        print("C, εのチューニング")
        print("最適なパラメーター =", gridsearch.best_params_)
        print("精度 =", gridsearch.best_score_)
        print()

        # 検証曲線
        plt_x, plt_y = np.meshgrid(params["C"], params["epsilon"])
        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.3)
        for i in range(2):
            if i == 0:
                plt_z = np.array(gridsearch.cv_results_["mean_train_score"]).reshape(params_cnt, params_cnt, order="F")
                title = "Train"
            else:
                plt_z = np.array(gridsearch.cv_results_["mean_test_score"]).reshape(params_cnt, params_cnt, order="F")
                title = "Cross Validation"
            ax = fig.add_subplot(2, 1, i + 1)
            CS = ax.contour(plt_x, plt_y, plt_z, levels=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
            ax.clabel(CS, CS.levels, inline=True)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("C")
            ax.set_ylabel("epsilon")
            ax.set_title(title)
        plt.suptitle("Validation curve / Gaussian SVR")
        plt.show()

        return gridsearch

    def do_fit(self, X_norm, y, gridsearch):
        # チューニングしたC,εでフィット
        regr = SVR(C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
        train_indices = next(self.gen_cv(y))[0]
        valid_indices = next(self.gen_cv(y))[1]
        regr.fit(X_norm[train_indices, :], y[train_indices])

        return regr, train_indices, valid_indices