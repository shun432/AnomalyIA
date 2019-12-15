import config as cfg

import sampledata as sd
import classify
import augment
import analysis
import Flist



class CImodel:


    def __init__(self, cfg, classifier):



        ############ parameter ############

        self.app_num = cfg.APP_NUM

        self.app = []
        self.trend_rule = None
        self.prediction = [[] for i in range(self.app_num)]
        self.classifier = classifier



        ############ setup ############

        import copy

        # アプリをn個用意
        for i in range(self.app_num):
            self.app.append(sd.ApplicationData())

        # アプリの特徴データを登録
        for i in range(self.app_num):
            j = 0
            for feature in cfg.DATATYPE:

                # vector["data"]のクラスインスタンスを参照渡しではなくディープコピーする
                self.app[i].addDataType(copy.deepcopy(feature["data"]), feature["name"])

                # FIRST_BINに従ってバイナリの初期値を設定する
                if type(feature) is sd.BinaryType:
                    if cfg.FIRST_BIN[i][j] == 1:
                        self.app[i].DataType.change()

                j += 1

        # トレンドルールを登録
        self.trend_rule = sd.TrendRule(cfg.FIRST_W, cfg.TYPE_LENGTH, delta=cfg.SHIFT_TREND_RULE)


    def update(self):

        # for app in apps:
            ## 各アプリについてトレンド予測（分類）を試みる
            ## 前シーズン行った予測に対して評価し、トレンド予測が上手くいったかを返す

        # for app in pred_failures:
            ## トレンド予測が上手くいかなかったアプリのデータオーグメントをする

        # for app in augmented:
            ## 集まったアプリをデータ分析（短期的で過学習な分析）（クラスタ？主成分？NN？）

        ## 分析結果から新トレンドルール群を予測

        return      ## 新トレンドルール群を返す






if __name__ == '__main__':

    classifier = classify.LSTMa(cfg.TYPE_LENGTH, cfg.LSTM_REFERENCE_STEPS)

    CIM = CImodel(cfg, classifier)

    for season in range(cfg.SPAN):

        new_trendrule = CIM.update()

        ### 時系列更新処理
        # for app in apps:
            ## 各アプリのデータを更新

        ## トレンドルールを更新


    # モデルの結果を出力

