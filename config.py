'''

        パラメータを設定するためのファイル

'''

import sampledata as sd

# アプリの数
APP_NUM = 5

# 全体の期間
SPAN = 100

# トレンドルールの毎シーズン更新度　（大きいところころルールが変わる）
SHIFT_TREND_RULE = 0.05


# アプリの特徴の形
DATATYPE = [dict(name="data1", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
            dict(name="data2", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
            dict(name="data3", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
            dict(name="data4", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
            dict(name="data5", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
            dict(name="data6", data=sd.ScalarNormType(mu=0, sigma=0.05, thread=True)),
            dict(name="genre1", data=sd.BinaryType(possibility=0)),
            dict(name="genre2", data=sd.BinaryType(possibility=0)),
            dict(name="genre3", data=sd.BinaryType(possibility=0)),
            dict(name="genre4", data=sd.BinaryType(possibility=0))]

# 各アプリのバイナリ型特徴の初期値      FIRST_BIN[アプリ][上からn個目の特徴]
FIRST_BIN = [[0, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 1, 1]]

# データタイプを増やすと横に広げる、縦に広げると並列ルールが増える      FIRST_W[ルール][特徴][時系列]
FIRST_W = [[[0.2], [0.5], [0.2], [-0.1], [0.0], [0.8], [0.2], [0.2], [0.5], [-0.3]],
           [[0.2], [0.5], [-0.5], [0.3], [0.5], [0.2], [-0.1], [0.0], [0.8], [0.2]],
           [[0.2], [0.2], [0.5], [-0.3], [0.5], [-0.5], [0.3], [0.5], [0.2], [-0.1]],
           [[0.8], [-0.5], [0.3], [0.2], [0.8], [0.8], [0.8], [0.3], [0.1], [0.3]],
           [[0.5], [0.2], [0.1], [0.8], [0.1], [-0.2], [0.2], [-0.9], [0.4], [-0.8]],
           [[0.8], [0.8], [0.1], [0.1], [-0.5], [0.1], [0.5], [0.6], [-0.5], [0.9]],
           [[0.0], [0.0], [0.8], [0.2], [0.2], [0.5], [-0.3], [0.5], [-0.5], [0.3]]]



# アプリの特徴次元数の受け取り用
TYPE_LENGTH = len(DATATYPE)



# LSTM用パラメータ
LSTM_REFERENCE_STEPS = 10
