import numpy as np
import math
import random
from matplotlib import pyplot


# 一次元の時系列データを作る
class OneDimTimeSeries:

    # データの単位時間
    DT = 1

    # 合計時間
    endTIME = 5000

    # 全体的な周波数(通常は１くらい)
    freq = 3

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
                if(size > self.Qabpt_min):
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

            abran_val, flag = self.abnormal_range_function(0.1, flag, A)
            abpt_val = self.abnormal_point_function(0.6)

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
        print(len(value))
        print(len(x))
        print(len(ano_value))
        pyplot.plot(x, value)
        pyplot.scatter(x, ano_value, marker=".", color="red")
        pyplot.show()

        return value, ano_value, label

if __name__ == '__main__':

    data, ano_data, _ = OneDimTimeSeries.make_value()

    x = list(range(len(ano_data)))
    print(len(data))
    print(len(x))
    print(len(ano_data))
    pyplot.plot(x, data)
    pyplot.scatter(x, ano_data, marker=".", color="red")
    pyplot.savefig("sample_data.png")
    pyplot.show()