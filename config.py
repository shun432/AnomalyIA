'''

        パラメータを設定するためのファイル

'''

import sampledata as sd
import random

# モデル進行状態の表示
SHOW_MODEL_DETAIL = True

# アプリの数
APP_NUM = 50

# 全体の期間
SPAN = 200

# アプリの流行値が明らかになるまでの期間
REVEAL_TREND = 5
if REVEAL_TREND <= 0:
    print("REVEAL_TREND is more than 0")

# トレンドルールの毎シーズン更新度　（大きいところころルールが変わる）
SHIFT_TREND_RULE = 0.07


# 分類器の予測誤差でデータを間引くための閾値
EVALUATE_THRESHOLD_PRED_FAIL = 0.3


# 分析器のアプリのサンプリング数
SAMPLING = 3

# 分析器のルール削除のための閾値（これ未満が０個で消去）
EVALUATE_THRESHOLD_DELETE_RULE = 0.2

# 分析器の新ルール適用のための閾値（これ未満で追加）
EVALUATE_THRESHOLD_ADD_RULE = 0.2

# 分析器のマージのための閾値（これ未満のアプリの組み合わせでマージ）
EVALUATE_THRESHOLD_MERGE_RULE = 0.2

# 分析器の新ルールが何個のアプリ以上で採用かの閾値
THRESHOLD_APPNUM = 3

# 分析器の新ルール作成試行回数
TRY_NEWRULE_NUM = 20


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
FIRST_W = [[[random.uniform(-1, 1)] for i in range(10)] for j in range(20)]

# アプリの特徴次元数の受け取り用
TYPE_LENGTH = len(DATATYPE)



# 分類器LSTM用パラメータ
LSTM_REFERENCE_STEPS = 100
LSTM_EPOCHS = 5



# 分析器NN用パラメータ
NN_EPOCHS = 20
