import sampledata as sd
import classify

from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import os
import time

'''
        モデルを実行する関数を置く場所
'''

# グラフを名前をつけて自動保存する関数
def save_fig(path):
    n = 1
    while True:
        if not os.path.exists(path + "_" + str(n) + ".png"):
            break
        else:
            n += 1
    plt.savefig(path + "_" + str(n) + ".png")


# 一次元データを扱う用
class run_classify_for_OD:

    def run_OCsvm(self):

        model = classify.OneClassSVM()

        data, ano_data, label, dataLen  = None, None, None, None

        for i in range(1):
            data, ano_data, label = sd.OneDimTimeSeries.make_value()
            dataLen = sd.OneDimTimeSeries.endTIME + 1

            a = [0] * dataLen
            data = np.array(data + a)
            data = data.reshape(2, dataLen).T

            model.dofit(data, label)

        pred = model.dopredict(data)

        #結果を出力しやすい形に変える
        for i in range(dataLen):
            if (pred[i] == 1):
                pred[i] = -15
            else:
                pred[i] = data[i][0]

        x = list(range(len(ano_data)))
        plt.plot(x, data)
        plt.scatter(x, pred, marker="o", color="green")
        plt.scatter(x, ano_data, marker=".", color="red")
        save_fig("result/OneDim/OCsvm/predict" + str(sd.OneDimTimeSeries.noise_type))
        plt.show()

    def run_knn(self, Threshold = 0.01):

        model = classify.kNN()

        data, ano_data, _ = sd.OneDimTimeSeries.make_value()
        dataLen = sd.OneDimTimeSeries.endTIME + 1


        #sklearnは2次元行列に変換しないと扱えない
        a = [0] * dataLen
        data = np.array(data + a)
        data = data.reshape(2, dataLen).T

        model.dofit(data)
        dist = model.dopredict(data)

        pred = []

        # plt.plot(dist)
        # plt.show()

        #結果を出力しやすい形に変える
        for i in range(dataLen):
            if (dist[i][1] < Threshold):
                pred.append(-15)
            else:
                pred.append(data[i][0])

        x = list(range(len(ano_data)))
        plt.plot(x, data)
        plt.scatter(x, pred, marker="o", color="green")
        plt.scatter(x, ano_data, marker=".", color="red")
        save_fig("result/OneDim/knn/predict" + str(sd.OneDimTimeSeries.noise_type))
        plt.show()

    def run_LPF(self, Threshold):

        lpf = classify.LPF()

        data, ano_data, _ = sd.OneDimTimeSeries.make_value()

        lowpassed = lpf.model(data)
        dataLen = sd.OneDimTimeSeries.endTIME + 1

        pred = []
        thLPF_o = []
        thLPF_u = []

        for i in range(dataLen):
            if (abs(lowpassed[i] - data[i]) < Threshold):
                pred.append(-15)
            else:
                pred.append(data[i])

            thLPF_o.append(lowpassed[i] + Threshold)
            thLPF_u.append(lowpassed[i] - Threshold)


        x = list(range(len(ano_data)))
        plt.plot(x, data)
        plt.plot(x, lowpassed)
        # plt.plot(x, thLPF_o, linestyle="dashed")
        # plt.plot(x, thLPF_u, linestyle="dashed")
        plt.scatter(x, pred, marker="o", color="green")
        plt.scatter(x, ano_data, marker=".", color="red")
        save_fig("result/OneDim/LPF/predict" + str(sd.OneDimTimeSeries.noise_type))
        plt.show()

    def run_LSTM(self, Threshold):

        lstm = classify.LSTMs()

        raw_data, _, label = sd.OneDimTimeSeries.make_value()
        data, target, label = lstm.preprocessing(np.array(raw_data), np.array(label))

        model = lstm.modeling()
        history, predicted = lstm.dofit(data, target, model, label)
        # future_result = lstm.predict_future(data, model)

        length = lstm.reference_len

        pred = []
        for i in range(len(predicted)):
            #labelを予測
            if (predicted[i] > Threshold):
                pred.append(-15)
            else:
                pred.append(raw_data[i+length-1])

            # #targetを予測
            # if (abs(predicted[i]*13 - raw_data[i+12]) < Threshold):
            #     pred.append(-15)
            # else:
            #     pred.append(raw_data[i+12])

        # Plot Wave
        plt.figure()
        plt.plot(range(0, len(raw_data)), raw_data, color="b", label="data")
        plt.plot(range(length, len(predicted) + length), predicted, color="r", label="rawOutput")
        plt.scatter(range(length, len(pred) + length), pred, color="g", label="RecogAnomaly")
        plt.scatter(range(length, len(label) + length), label, label, color="y", label="TruthAnomaly")
        # plt.plot(range(0 + len(raw_data) - 24, len(future_result) + len(raw_data) - 24), future_result, color="orange",
        #          label="future")
        plt.legend()
        save_fig("result/OneDim/lstm/predict" + str(sd.OneDimTimeSeries.noise_type))
        plt.show()
        plt.close()

        # Plot Training loss & Validation Loss
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and Validation loss")
        plt.legend()
        save_fig("result/OneDim/lstm/loss" + str(sd.OneDimTimeSeries.noise_type))
        plt.close()

    def run_AutoEncoder(self, Threshold):

        autoencoder = classify.AutoEncoder()

        raw_data, ano_data, label = sd.OneDimTimeSeries.make_value()
        data, target = autoencoder.preprocessing(np.array(raw_data))

        model = autoencoder.modeling()
        history, predicted = autoencoder.dofit(data, model)

        length = autoencoder.reference_len

        pred = []
        offset = autoencoder.offset

        for i in range(len(predicted)):
            # #labelを予測
            # if (predicted[i] > 0):
            #     pred.append(-15)
            # else:
            #     pred.append(raw_data[i+length])

            #targetを予測
            if (abs((predicted[i][length-1]*offset-offset/2) - raw_data[i+length-1]) < Threshold):
                pred.append(-15)
            else:
                pred.append(raw_data[i+length-1])

        # Plot Wave
        plt.figure()
        plt.plot(range(0, len(raw_data)), raw_data, color="b", label="data")
        plt.scatter(range(length, len(pred) + length), pred, marker="o", color="g", label="anomaly")
        plt.scatter(range(0, len(ano_data)), ano_data, marker=".", color="r", label="TruthAnomaly")
        # plt.plot(range(0 + len(raw_data) - 24, len(future_result) + len(raw_data) - 24), future_result, color="orange",
        #          label="future")
        plt.legend()
        save_fig("result/OneDim/AutoEncoder/predict" + str(sd.OneDimTimeSeries.noise_type))
        plt.show()
        plt.close()

        # Plot Training loss & Validation Loss
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and Validation loss")
        plt.legend()
        save_fig("result/OneDim/AutoEncoder/loss" + str(sd.OneDimTimeSeries.noise_type))
        plt.close()

    def run_SSA(self, s=2):

        ssa = classify.SSA()

        sst, ano_data, label = sd.OneDimTimeSeries.make_value()
        sst = np.array(sst)

        ano = [[]]

        start = time.time()

        score = ssa.SSA_CD(series=sst,
                                    standardize=True,
                                    w=70, lag=24, ns_h=s, ns_t=1)

        elapsed_time = time.time() - start

        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        ano.append(score[0])
        x = list(range(len(sst)))

        plt.plot(x, sst)
        plt.plot(list(range(len(ano[1]))), np.array(ano[1]) * 150 - 15, color="green")
        plt.scatter(x, ano_data, marker=".", color="red")
        save_fig("result/OneDim/SSA/predict" + str(sd.OneDimTimeSeries.noise_type))
        plt.show()


        # for s in range(1, 3):
        #     score = classify.SSA.SSA_CD(series=sst,
        #                                 standardize=False,
        #                                 w=40, lag=24, ns_h=s, ns_t=1)
        #     ano.append(score[0])
        #
        # x = list(range(len(sst)))
        # for i in range(1, 3):
        #     plt.plot(x, sst)
        #     plt.plot(list(range(len(ano[i]))), np.array(ano[i]) * 8 - 15, color="green")
        #     plt.scatter(x, ano_data, marker=".", color="red")
        #     save_fig("SSA")
        #     plt.show()

    def run_SVR(self):

        # from sklearn.datasets import load_boston
        # from sklearn.model_selection import train_test_split
        # boston = load_boston()
        # X, Xtest, y, ytest = train_test_split(boston['data'], boston['target'], test_size=0.2, random_state=114514)
        #
        # print(np.shape(X))

        svr = classify.SVRs()

        # 訓練データ、テストデータに分割
        data, ano_data, label = sd.OneDimTimeSeries.make_value()

        data = np.array(data)
        label = np.array(label)

        x = data.reshape(-1, 1)

        # (それぞれ303 101 102 = サンプル合計は506)
        print("ガウシアンカーネルのSVR")
        print()

        data_norm = svr.preprocessing(x)

        print("1")

        gridsearch = svr.tuning(data_norm, label)

        print("2")

        regr, train_indices, valid_indices = svr.do_fit(data_norm, label, gridsearch)

        # テストデータの精度を計算
        print("テストデータにフィット")
        # print("テストデータの精度 =", regr.score(Xtest_norm, ytest))
        # print()
        print("※参考")
        print("訓練データの精度 =", regr.score(data_norm[train_indices, :], label[train_indices]))
        print("交差検証データの精度 =", regr.score(data_norm[valid_indices, :], label[valid_indices]))

        predicted = regr.predict(data_norm)

        x = list(range(len(ano_data)))
        plt.plot(x, data)
        plt.scatter(x, predicted, marker="o", color="green")
        plt.scatter(x, ano_data, marker=".", color="red")
        save_fig("result/OneDim/SVR/predict" + str(sd.OneDimTimeSeries.noise_type))
        plt.show()

    def run(self, select):
        if select == "knn":
            self.run_knn(0.002)
        elif select == "ocsvm":
            self.run_OCsvm()
        elif select == "lpf":
            self.run_LPF(1.5)
        elif select == "lstm":
            self.run_LSTM(0.3)
        elif select == "ae":
            self.run_AutoEncoder(0.6)
        elif select == "ssa":
            self.run_SSA(2)
        elif select == "svr":
            self.run_SVR()
        else:
            print("set correct algorithm name")


# アプリの特徴ベクトルのような多次元データを扱う用
class run_classify_for_App:

    def __init__(self, n=5, span=100, DataType=None, first_w=None, w_delta=0.05):

        # parameter

        self.span = span
        self.w_delta = w_delta

        # setup

        self.app = []
        self.prediction = [[] for i in range(n)]

        for i in range(n):
            self.app.append(sd.ApplicationData())

        # データを用意してself.appとself.trend_ruleに格納
        for i in range(n):
            self.prepare_data(self.app[i], DataType[i])

        self.trend_rule = None
        self.init_trend(first_w, self.w_delta)

        # for i in range(100):
        #     app.update(self.trend_rule)
        #     self.trend_rule.update()

    def prepare_data(self, app, DataType):

        if DataType is None:
            # デフォルトのデータタイプ
            app.addDataType(sd.ScalarNormType(mu=0, sigma=0.05, thread=True), "Norm")
            app.addDataType(sd.ScalarNormType(mu=0, sigma=0.05, thread=True), "Norm")
            app.addDataType(sd.ScalarNormType(mu=0, sigma=0.05, thread=True), "Norm")
            app.addDataType(sd.BinaryType(possibility=0, initval=0), "Bina")
        else:
            for feature in DataType:
                app.addDataType(feature["data"], feature["name"])

    def init_trend(self, first_w, w_delta):

        w = first_w

        if first_w is None:

            # デフォルトのルール重み
            w = [[[0.2], [0.5], [0.2], [-0.1]],
                 [[0.2], [0.5], [-0.5], [0.3]],
                 [[0.2], [0.2], [0.5], [-0.3]],
                 [[0.0], [0.0], [0.8], [0.2]]]

        self.trend_rule = sd.TrendRule(w, self.app[0].typelength, delta=w_delta)

    def savefig_result(self, start_offset=0):

        x = list(range(self.span))

        plt.figure(figsize=(len(x)/10, 5))

        # アプリごとの色
        if len(self.app) <= 10:
            cycle_app = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif len(self.app) <= 20:
            cycle_app = plt.cm.get_cmap('tab20')
        else:
            cycle_app = list(colors.XKCD_COLORS.items())[:100]

        for id in range(len(self.app)):
            plt.plot(x, self.app[id].trend, color=cycle_app[id], label="trend (app:" + str(id) + ")", linestyle="dotted")
            plt.plot(x[start_offset:], self.prediction[id], color=cycle_app[id], label="pred (app:" + str(id) + ")")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.8)
        save_fig("result/AppDATA/PredictTrend")

        plt.clf()

        return

    def savefig_ruleweight(self):

        x = list(range(self.span))

        plt.figure(figsize=(len(x)/10, 5))

        # トレンドルールごとの色
        if len(self.trend_rule.w) <= 10:
            cycle_tr = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif len(self.trend_rule.w) <= 20:
            cycle_tr = plt.cm.get_cmap('tab20')
        else:
            cycle_tr = list(colors.XKCD_COLORS.items())[:100]

        # 特徴ごとの色
        if len(self.trend_rule.w[0]) <= 10:
            cycle_ft = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif len(self.trend_rule.w[0]) <= 20:
            cycle_ft = plt.cm.get_cmap('tab20')
        else:
            cycle_ft = list(colors.XKCD_COLORS.items())[:100]

        width = 0.8 / len(self.trend_rule.w[0])
        #トレンドルール毎に
        for i in range(len(self.trend_rule.w)):
            bottom = np.array(- i * 2.0)
            # 特徴毎に
            for j in range(len(self.trend_rule.w[i])):
                if i == 0:
                    plt.bar(x + np.array([width * float(j)] * len(x)), self.trend_rule.w[i][j][:-1],
                            color=cycle_ft[j], align='edge', bottom=bottom, width=width, label="feature:" + str(j))
                else:
                    plt.bar(x + np.array([width * float(j)] * len(x)), self.trend_rule.w[i][j][:-1],
                            color=cycle_ft[j], align='edge', bottom=bottom, width=width)

            plt.fill_between(list(range(self.span+1)), [- i * 2.0 + 1] * (len(x)+1), [- (i+1) * 2.0 + 1] * (len(x)+1),
                             facecolor=cycle_tr[i], alpha=0.2, label="trendrule:" + str(i))

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.8)
        save_fig("result/AppDATA/TrendRuleW")


        plt.clf()

        return

    def savefig_chosenrule(self):

        x = list(range(self.span))

        plt.figure(figsize=(len(x)/10, 5))

        # アプリごとの色
        if len(self.app) <= 10:
            cycle_app = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif len(self.app) <= 20:
            cycle_app = plt.cm.get_cmap('tab20')
        else:
            cycle_app = list(colors.XKCD_COLORS.items())[:100]

        # トレンドルールごとの色
        if len(self.trend_rule.w) <= 10:
            cycle_tr = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif len(self.trend_rule.w) <= 20:
            cycle_tr = plt.cm.get_cmap('tab20')
        else:
            cycle_tr = list(colors.XKCD_COLORS.items())[:100]

        # 凡例表示用
        for i in range(len(self.trend_rule.w)):
            plt.scatter(x, np.array([0] * len(x)), color=cycle_tr[i], s=1, marker="D",
                        label="trendrule:" + str(i))

        for id in range(len(self.app)):
            colorArr = []
            for i in self.app[id].trend_idx:
                colorArr.append(cycle_tr[i])
            plt.scatter(x, np.array([- id] * len(x)), color=cycle_app[id], s=150, label="app:" + str(id))
            plt.scatter(x, np.array([- id] * len(x)), color="w", s=70)
            plt.scatter(x, np.array([- id] * len(x)), color=colorArr, s=15, marker="D", alpha=0.5)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.8)
        save_fig("result/AppDATA/ChosenRule")

        plt.clf()

        return

    def run_OCsvm(self):

        return

    def run_LSTM(self, reference_steps=10):

        rs = reference_steps
        tl = self.app[0].typelength

        lstm = classify.LSTMa(tl, rs, epochs=2)

        # 各時系列で
        for t in range(self.span):

            print("time:" + str(t) + "/" + str(self.span - 1))

            # それぞれのアプリについて
            for app_id in range(len(self.app)):

                data = None

                if t >= rs:

                    data = np.array([u[-rs:] for u in self.app[app_id].featureVector]).T.reshape(rs, tl, 1)

                    # 最新データからトレンドを予測する
                    pred = lstm.dopredict(data)

                    # 最新データの予測のみ格納
                    self.prediction[app_id].append(pred[-1][0])

                if t >= rs + 1:
                    # 前シーズンまでの特徴を学習する
                    history = lstm.dofit(data, np.array(self.app[app_id].trend[-(rs+1):-1]).reshape(rs, 1))

                self.app[app_id].update(self.trend_rule)

            self.trend_rule.update()

        # データを保存する
        self.savefig_result(start_offset=rs+2)
        self.savefig_ruleweight()
        self.savefig_chosenrule()

        return


if __name__ == '__main__':

    # runOD = run_classify_for_OD()
    # runOD.run("ae")

    app_num = 5

    first_bin = [[0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 1, 1]]

    DataType = []
    for i in range(app_num):
        DataType.append(
            [dict(name="data1", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
             dict(name="data2", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
             dict(name="data3", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
             dict(name="data4", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
             dict(name="data5", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
             dict(name="data6", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
             dict(name="genre1", data=sd.BinaryType(possibility=0, initval=first_bin[i][0])),
             dict(name="genre2", data=sd.BinaryType(possibility=0, initval=first_bin[i][1])),
             dict(name="genre3", data=sd.BinaryType(possibility=0, initval=first_bin[i][2])),
             dict(name="genre4", data=sd.BinaryType(possibility=0, initval=first_bin[i][3]))])

    # データタイプを増やすと横に広げる、縦に広げると並列ルールが増える      w[ルール][特徴][時系列]
    first_w = [[[0.2], [0.5], [0.2], [-0.1], [0.0], [0.8], [0.2], [0.2], [0.5], [-0.3]],
               [[0.2], [0.5], [-0.5], [0.3], [0.5], [0.2], [-0.1], [0.0], [0.8], [0.2]],
               [[0.2], [0.2], [0.5], [-0.3], [0.5], [-0.5], [0.3], [0.5], [0.2], [-0.1]],
               [[0.8], [-0.5], [0.3], [0.2], [0.8], [0.8], [0.8], [0.3], [0.1], [0.3]],
               [[0.5], [0.2], [0.1], [0.8], [0.1], [-0.2], [0.2], [-0.9], [0.4], [-0.8]],
               [[0.8], [0.8], [0.1], [0.1], [-0.5], [0.1], [0.5], [0.6], [-0.5], [0.9]],
               [[0.0], [0.0], [0.8], [0.2], [0.2], [0.5], [-0.3], [0.5], [-0.5], [0.3]]]

    run = run_classify_for_App(n=app_num, DataType=DataType, first_w=first_w)
    run.run_LSTM()
