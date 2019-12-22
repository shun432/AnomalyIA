'''

        パラメータを設定するためのファイル

'''

import sampledata as sd

# モデル進行状態の表示
SHOW_MODEL_DETAIL = True

# アプリの数
APP_NUM = 10

# 全体の期間
SPAN = 100

# トレンドルールの毎シーズン更新度　（大きいところころルールが変わる）
SHIFT_TREND_RULE = 0.05


# 分類器の予測誤差でデータを間引くための閾値
EVALUATE_THRESHOLD_PRED_FAIL = 0.2


# 分析器のアプリのサンプリング数
SAMPLING = 3

# 分析器のルール削除のための閾値
EVALUATE_THRESHOLD_DELETE_RULE = 0.3

# 分析器の新ルール適用のための閾値
EVALUATE_THRESHOLD_ADD_RULE = 0.2

# 分析器の新ルールが何個のアプリ以上で採用かの閾値
THRESHOLD_APPNUM = 3


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



# 分類器LSTM用パラメータ
LSTM_REFERENCE_STEPS = 10
LSTM_EPOCHS = 2



# 分析器NN用パラメータ
NN_EPOCHS = 20




