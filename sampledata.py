import numpy as np
import math
import random
from matplotlib import pyplot as plt


# 一次元の時系列データを作る
class OneDimTimeSeries:

    # データの単位時間
    DT = 1

    # 合計時間
    endTIME = 5000

    # 全体的な周波数(通常は１くらい)
    freq = 3

    #異常確率
    Pabpt = 0.3
    Pabran = 0.05

    # 各ノイズの大きさ
    # 正常ノイズ
    Q = 0.1
    # 外れ値ノイズの最大値、最小値
    Qabpt_max = 5.0
    Qabpt_min = 2.0
    # 異常部位ノイズの最大値、最小値
    Qabran_max = 3.0
    Qabran_min = 2.0

    # どのノイズを乗せるか( 0 or 1 or other )
    noise_type = 1

    if(noise_type == 0):
        abpt = 1
        abran = 0
    elif(noise_type == 1):
        abpt = 0
        abran = 1
    else:
        abpt = 1
        abran = 1

    # 引数％の確率で異常部位を乗せる
    @classmethod
    def abnormal_range_function(self, probability, time, A):

        # 異常部位の長さを決定
        range = self.DT * (150 + random.uniform(-100, 100))

        if probability * 0.01 > random.random() or time >= self.DT:
            time += self.DT
            A *= math.sin(math.radians(time) * self.freq)
            if time >= range and abs(A) < 0.01:
                return A, 0
            else:
                return A, time

        return 0, 0

    # 引数％の確率で外れ値を乗せる
    @classmethod
    def abnormal_point_function(self, probability):
        if probability * 0.01 > random.random():
            while (True):
                size = self.Qabpt_max * random.uniform(-1, 1)
                if(abs(size) > self.Qabpt_min):
                    return size
        return 0

    # ベースとなる関数を指定
    @classmethod
    def normal_function(self, time):
        rad_t = math.radians(time)
        return 5.0 * math.sin(0.3 * rad_t * self.freq)



    # これを外部から呼ぶと配列で時系列データを返す
    @classmethod
    def make_value(self):

        value = []
        ano_value = []
        label = []

        time = 0.0

        flag = 0

        while(time <= self.endTIME):

            if(flag == 0):
                while(True):
                    A = self.Qabran_max * random.uniform(-1, 1)
                    if(abs(A) > self.Qabran_min):
                        break

            abran_val, flag = self.abnormal_range_function(self.Pabran, flag, A)
            abpt_val = self.abnormal_point_function(self.Pabpt)

            value.append(self.normal_function(time) + np.random.randn() * self.Q
                         + self.abpt * abpt_val + self.abran * abran_val)

            if(self.abran * abran_val!=0 or self.abpt * abpt_val!=0):      # 異常
                ano_value.append(value[-1])
                label.append(-1)
            else:                          # 正常
                ano_value.append(-15)
                label.append(1)

            time += self.DT

        x = list(range(len(ano_value)))
        plt.plot(x, value)
        plt.scatter(x, ano_value, marker=".", color="red")
        plt.show()

        return value, ano_value, label



# アプリのサンプルデータ
class ApplicationData:

    # アプリの特徴ベクトルなどのパラメータ
    def __init__(self):

        self.typelength = 0
        self.DataName = []
        self.DataType = []
        self.featureVector = []         # featureVector[特徴][時系列]
        self.trend = []
        self.trend_idx = []

    # 特徴ベクトルの基底を追加（収益、DL数、カテゴリーなど）
    def addDataType(self, type, name="NoNamed"):

        self.typelength += 1
        self.DataName.append(name)
        self.DataType.append(type)
        self.featureVector.append([])

    # 時系列を進める
    def update(self, trend_rule):

        newdata = []

        for i in range(self.typelength):
            newdata.append(self.DataType[i].makedata())
            self.featureVector[i].append(newdata[i])

        value, applied_idx = trend_rule.apply(newdata)
        self.trend.append(value)
        self.trend_idx.append(applied_idx)


# 一様分布のスカラー乱数を生成（thread=Trueで直前のデータに上乗せ）
class ScalarNormType:

    def __init__(self, mu, sigma, thread=False):

        self.mu = mu
        self.sigma = sigma
        self.thread = thread
        self.lastdata = 0

    def makedata(self):

        if not self.thread:
            return random.normalvariate(self.mu, self.sigma)
        else:
            self.lastdata += random.normalvariate(self.mu, self.sigma)
            return self.lastdata


# 一様分布のスカラー乱数を生成（thread=Trueで直前のデータに上乗せ）
class ScalarUnifType:

    def __init__(self, min, max, thread=False):

        self.min = min
        self.max = max
        self.thread = thread
        self.lastdata = 0

    def makedata(self):

        if not self.thread:
            return random.uniform(self.min, self.max)
        else:
            self.lastdata += random.uniform(self.min, self.max)
            return self.lastdata


# バイナリデータを生成（値の変化確率possibilityは0~1）
class BinaryType:

    def __init__(self, possibility=0.0, initval=0):

        self.lastdata = initval
        self.possibility = possibility

    def change(self):

        if self.lastdata is 0:
            self.lastdata = 1
        else:
            self.lastdata = 0

    def makedata(self):

        if random.random() < self.possibility:
            self.change()

        return self.lastdata


# 流行のルールを管理する。どの特徴を重視するかをwで調節。複数のルールをwに格納できる
class TrendRule:

    def __init__(self, init_w, datalen, delta=0.1):

        self.w = init_w
        self.datalen = datalen
        self.delta = delta

    # ルールを適用して流行度合いを返す
    def apply(self, currentdata):

        val_max = 0
        applied_idx = 0

        for i in range(len(self.w)):
            val = 0
            for j in range(self.datalen):
                val += self.w[i][j][-1] * currentdata[j]
            if val > val_max:
                applied_idx = i
                val_max = val

        return val_max, applied_idx

    # ルールを更新する
    def update(self):

        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                w = self.w[i][j][-1] + self.delta * random.uniform(-1, 1)
                self.w[i][j].append(min(max(w, -1.0), 1.0))



if __name__ == '__main__':

    app1 = ApplicationData()

    app1.addDataType(ScalarNormType(mu=0, sigma=0.05, thread=True), "Norm")
    app1.addDataType(ScalarNormType(mu=0, sigma=0.05, thread=True), "Norm")
    app1.addDataType(ScalarNormType(mu=0, sigma=0.05, thread=True), "Norm")
    app1.addDataType(BinaryType(possibility=0, initval=1), "Bina")

    # データタイプを増やすと横に広げる、縦に広げると並列ルールが増える
    init_w = [[[ 0.2], [ 0.5], [ 0.2], [-0.1]],
              [[ 0.2], [ 0.5], [-0.5], [ 0.3]],
              [[ 0.2], [ 0.2], [ 0.5], [-0.3]],
              [[ 0.0], [ 0.0], [ 0.8], [ 0.2]]]

    trend_rule = TrendRule(init_w, app1.typelength, delta=0.05)

    for i in range(100):
        app1.update(trend_rule)
        trend_rule.update()

    print("create data completed...")

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    x = list(range(100))
    for i in range(app1.typelength):
        plt.plot(x, app1.featureVector[i], color=cycle[i], label=app1.DataName[i])
    plt.legend()
    plt.savefig("app1_data.png")

    plt.show()

    plt.clf()

    width = 0.8 / len(trend_rule.w)
    for i in range(len(trend_rule.w)):
        for j in range(len(trend_rule.w[i])):
            bottom = np.array(i * 2.0)
            plt.bar(x + np.array([width * float(j)]*len(x)), trend_rule.w[i][j][:-1],
                    color=cycle[j], align='edge', bottom=bottom, width=width)

    plt.bar(x, [0.05] * len(x), color="blue", align='edge', bottom=np.array(app1.trend_idx)*2, width=0.8)
    plt.plot(x, app1.trend[:], color="black")

    plt.savefig("app1_ruleweight.png")

    plt.show()


    # data, ano_data, _ = OneDimTimeSeries.make_value()
    #
    # x = list(range(len(ano_data)))
    # plt.plot(x, data)
    # plt.scatter(x, ano_data, marker=".", color="red")
    # plt.savefig("sample_data.png")
