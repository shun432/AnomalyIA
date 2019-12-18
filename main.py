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

        self.apps = []
        self.trend_rule = None
        self.prediction = [[] for i in range(self.app_num)]
        self.history = [[] for i in range(self.app_num)]

        self.classifier = classifier
        self.rs = classifier.reference_steps
        self.dd = classifier.data_dimension



        ############ setup ############

        import copy

        # アプリをn個用意
        for i in range(self.app_num):
            self.apps.append(sd.ApplicationData())

        # アプリの特徴データを登録
        for i in range(self.app_num):
            j = 0
            for feature in cfg.DATATYPE:

                # vector["data"]のクラスインスタンスを参照渡しではなくディープコピーする
                self.apps[i].addDataType(copy.deepcopy(feature["data"]), feature["name"])

                # FIRST_BINに従ってバイナリの初期値を設定する
                if type(feature) is sd.BinaryType:
                    if cfg.FIRST_BIN[i][j] == 1:
                        self.apps[i].DataType.change()

                j += 1

        # トレンドルールを登録
        self.trend_rule = sd.TrendRule(cfg.FIRST_W, cfg.TYPE_LENGTH, delta=cfg.SHIFT_TREND_RULE)


    # CIMの1シーズン分を実行
    def run(self, season):

        # 各アプリについてトレンド予測（分類）を試みる
        if season >= self.rs:
            self.run_classify(season)

        # 前シーズン行った予測に対して評価し、トレンド予測が上手くいったかを返す
        if season >= self.rs + 1:

            self.learn_classify()
            self.evaluate_classify()



        # for app in pred_failures:
            ## トレンド予測が上手くいかなかったアプリのデータオーグメントをする

        # for app in augmented:
            ## 集まったアプリをデータ分析（短期的で過学習な分析）（クラスタ？主成分？NN？）

        ## 分析結果から新トレンドルール群を予測

        return      ## 新トレンドルール群を返す


    # 分類器で流行予測する
    def run_classify(self, season):

        for i, app in enumerate(self.apps):

            # 分類に前処理が必要であれば実行する
            try:
                # 今シーズンから過去10個分を切り出し
                data, _ = self.classifier.preprocessing(app.featureVector)
            except:
                data = app.featureVector

            # 最新データからトレンドを予測する
            pred = self.classifier.dopredict(data)

            # 最新データの予測のみ格納
            self.prediction[i].append(pred[-1][0])


    # 分類器の学習をする
    def learn_classify(self):

        for i, app in enumerate(self.apps):

            # 分類に前処理が必要であれば実行する
            try:
                # 前シーズンから過去10個分を切り出し
                data, target = self.classifier.preprocessing(app.featureVector[:-1], app.trend[:-1])
            except:
                data, target = app.featureVector[:-1], app.trend[:-1]

            # 前シーズンまでのトレンド結果をターゲットにして学習する
            self.history[i].append(self.classifier.dofit(data, target))


    # 前シーズンの予測結果を振り返る
    def evaluate_classify(self):


        return








if __name__ == '__main__':

    classifier = classify.LSTMa(cfg.TYPE_LENGTH, cfg.LSTM_REFERENCE_STEPS, epochs=2)

    CIM = CImodel(cfg, classifier)

    for season in range(cfg.SPAN):

        new_trendrule = CIM.run(season)

        ### 時系列更新処理
        # for app in apps:
            ## 各アプリのデータを更新

        ## トレンドルールを更新


    # モデルの結果を出力

