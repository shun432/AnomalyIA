import config as cfg

import sampledata as sd
import classify
import augment
import analysis
from figure import figure
import random
import time


# データを保存するクラス
class DataStore:

    def __init__(self, cfg):

        self.app_num = cfg.APP_NUM


        ############ data ############

        self.apps = []
        self.trend_rule = None


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
        self.trend_rule = sd.TrendRule(cfg.FIRST_RULE_NUM, cfg.TYPE_LENGTH, cfg.APPEAR_RATE, cfg.DISAPPEAR_RATE, delta=cfg.SHIFT_TREND_RULE)


# CIモデル
class CImodel:


    def __init__(self, cfg, classifier):


        ############ parameter ############

        self.app_num = cfg.APP_NUM

        self.classifier = classifier
        self.rs = classifier.reference_steps
        self.dd = classifier.data_dimension


        ############ data ############

        self.prediction = [[] for i in range(self.app_num)]
        self.history = [[] for i in range(self.app_num)]

        self.prediction_est_rule = [[] for i in range(self.app_num)]
        self.Estimated_rules = []

        self.prediction_only_ci = [[] for i in range(self.app_num)]  # *****************

        self.predfail_app_num = []
        self.rule_num = []
        self.add_rule_num = []
        self.lost_rule_num = []
        self.useless_rule_num = []
        self.merge_rule_num = []


    # CIMの1シーズン分を実行
    def run(self, apps, season):

        # 各アプリについてトレンド予測する
        if season >= self.rs + cfg.REVEAL_TREND:
            self.classify_predict(apps)

        # 前シーズンの予測の評価・分析
        if season >= self.rs + cfg.REVEAL_TREND + 1:

            start = time.time()

            # 分類器の学習
            self.classify_learn(apps)

            # 前シーズンの予測評価、上手くいかなかったアプリを返す
            predict_failure_apps, failed_idx = self.classify_evaluate_previous(apps, cfg.EVALUATE_THRESHOLD_PRED_FAIL)
            self.predfail_app_num.append(len(predict_failure_apps))
            if cfg.SHOW_MODEL_DETAIL:
                print("LSTMの予測失敗アプリ数 :" + str(len(predict_failure_apps)))

            # トレンド予測が上手くいかなかったアプリのデータオーグメントをする（仮）
            augmented_apps = predict_failure_apps

            # トレンド予測が上手くいかなかったアプリの分析・分析結果から予測
            self.analyse_failure(augmented_apps, failed_idx, apps)

            self.rule_num.append(len(self.Estimated_rules))
            if cfg.SHOW_MODEL_DETAIL:
                print("捕捉ルール数：" + str(len(self.Estimated_rules)))

            end = time.time()

            if cfg.SHOW_MODEL_DETAIL:
                print("処理時間：" + str(end - start))


    # 分類器で流行予測する
    def classify_predict(self, apps):

        for i, app in enumerate(apps):

            # 分類に前処理が必要であれば実行する
            try:
                # 今シーズンから過去10個分を切り出し
                data, _ = self.classifier.preprocessing(app.featureVector[:])
            except:
                data = app.featureVector
                print("except is called in classify_predict()")

            # 最新データからトレンドを予測する
            pred = self.classifier.dopredict(data)

            # 最新データの予測のみ格納
            self.prediction[i].append(pred[-1][0])


    # 分類器の学習をする
    def classify_learn(self, apps):

        for i, app in enumerate(apps):

            # 分類に前処理が必要であれば実行する
            try:
                # 前シーズンから過去10個分を切り出し
                data, target = self.classifier.preprocessing([u[:-1] for u in app.featureVector[:]], app.trend[:-1])
            except:
                data, target = app.featureVector[:-1], app.trend[:-1]
                print("except is called in classify_learn()")

            # 前シーズンまでのトレンド結果をターゲットにして学習する
            self.history[i].append(self.classifier.dofit(data, target))


    # 前シーズンの予測結果を振り返る
    def classify_evaluate_previous(self, apps, Threshold):

        predict_failure_apps = []
        idx = []

        for i, app in enumerate(apps):

            # 前シーズンのアプリトレンド値と予測結果を比べる
            if abs(app.trend[-2] - self.prediction[i][-2]) > Threshold:
                predict_failure_apps.append(app)
                idx.append(i)

        return predict_failure_apps, idx


    # 予測の失敗を分析する（分析関数の集合）
    def analyse_failure(self, failed_apps, failed_idx, apps):

        # 現存する推定ルールの見直し
        self.check_existing_rule(failed_apps)

        # 新ルールの捕捉を試みる
        new_rule_num = 0
        for i in range(cfg.TRY_NEWRULE_NUM):
            if self.capture_new_rule(failed_apps, apps):
                new_rule_num += 1
        self.add_rule_num.append(new_rule_num)
        if cfg.SHOW_MODEL_DETAIL:
            print("追加された新ルール数：" + str(new_rule_num))

        if len(self.Estimated_rules) > 0:

            # 各アプリに対する新旧ルールを評価し最適ルールを抽出する
            optimal_rule = self.counseling(apps)  # **********************実験用の変更点（apps→failed_appsに変更してください）

            # 推定ルールを使って流行予測
            self.predict_with_est_rule(failed_apps, failed_idx, apps, optimal_rule)

            # 全く同じアプリにしか通用しないルール同士があれば高性能な方にマージする
            self.merge_rule(failed_apps, optimal_rule)

        else:
            for i, app in enumerate(apps):
                self.prediction_est_rule[i].append(self.prediction[i][-1])
                self.prediction_only_ci[i].append(self.prediction[i][-1])
            self.merge_rule_num.append(0)


    # 既存の推定ルールを見直す
    def check_existing_rule(self, pred_fail_apps):

        lost_rule_num = 0
        useless_rule_num = 0

        for rule_id, estimated_rule in enumerate(self.Estimated_rules):

            if estimated_rule["status"] is not "new":

                num = 0
                for app in pred_fail_apps:

                    # 分析に前処理が必要であれば実行する
                    try:
                        # 前シーズンから過去10個分を切り出し
                        data, _ = estimated_rule["rule"].preprocessing([u[-cfg.REVEAL_TREND-2] for u in app.featureVector[:]])
                    except:
                        data = app.featureVector[-cfg.REVEAL_TREND-2]
                        print("except is called in check_existing_rule()")

                    # 既存のルールを前シーズンのデータで予測して評価する
                    evaluate = abs(estimated_rule["rule"].dopredict(data) - app.trend[-2])[0][0]

                    # このルールによるロスが閾値未満のアプリ数
                    if evaluate < cfg.EVALUATE_THRESHOLD_DELETE_RULE:
                        num += 1

                # 不要なルールは削除
                if num == 0:

                    if estimated_rule["status"] is "disappointed":
                        self.Estimated_rules.remove(estimated_rule)
                        lost_rule_num += 1
                        if cfg.SHOW_MODEL_DETAIL:
                            print("消去したルール:" + str(rule_id))

                    elif estimated_rule["status"] is "":
                        estimated_rule["status"] = "disappointed"
                        useless_rule_num += 1
                        if cfg.SHOW_MODEL_DETAIL:
                            print("不使用のルール:" + str(rule_id))

                elif estimated_rule["status"] is "disappointed":
                    estimated_rule["status"] = ""

            if estimated_rule["status"] is "new":
                estimated_rule["status"] = ""

        self.lost_rule_num.append(lost_rule_num)
        self.useless_rule_num.append(useless_rule_num)


    # 新しくルールの捕捉を試みる
    def capture_new_rule(self, pred_fail_apps, apps):

        # 新たなルール捕捉器を用意
        analyser = analysis.NeuralNetwork(cfg.TYPE_LENGTH, epochs=cfg.NN_EPOCHS)

        # アプリをランダムに数個選ぶ (cfg.SAMPLING個)
        sample_apps = random.sample(pred_fail_apps, len(pred_fail_apps))[:cfg.SAMPLING]

        # サンプルされたアプリを学習させる
        for app in sample_apps:

            # 分析に前処理が必要であれば実行する
            try:
                # 前シーズンから過去10個分を切り出し
                data, target = analyser.preprocessing([u[-cfg.REVEAL_TREND-2] for u in app.featureVector[:]], app.trend[-2])
            except:
                data, target = app.featureVector[-cfg.REVEAL_TREND-2], app.trend[-2]
                print("except is called in capture_new_rule1()")

            analyser.dofit(data, target)

        num = 0
        # 学習結果を他全てのアプリ（予測に失敗していないアプリを含む）で試す
        for app in apps:

            # 新ルール生成に用いたサンプルアプリ以外にフォーカス
            if not any([app is sample_app for sample_app in sample_apps]):

                # 分析に前処理が必要であれば実行する
                try:
                    # 前シーズンから過去10個分を切り出し
                    data, _ = analyser.preprocessing([u[-cfg.REVEAL_TREND-2] for u in app.featureVector[:]])
                except:
                    data = app.featureVector[-cfg.REVEAL_TREND-2]
                    print("except is called in capture_new_rule2()")

                evaluate = abs(analyser.dopredict(data) - app.trend[-2])[0][0]

                # if cfg.SHOW_MODEL_DETAIL:
                #     print("evaluate analyser :" + str(evaluate))

                # このルールによるロスが閾値未満のアプリ数
                if evaluate < cfg.EVALUATE_THRESHOLD_ADD_RULE:
                    num += 1

        # いくつかの他アプリでもルールが成り立ち、且つ他全てのアプリと成り立たつ訳でなければ新ルールとして登録
        if cfg.THRESHOLD_APPNUM < num < self.app_num - cfg.SAMPLING:
            self.Estimated_rules.append({"status": "new", "rule": analyser})
            return True
        else:
            return False


    # 全ての分析器を前シーズンで評価し各アプリの最適ルールを抽出
    def counseling(self, pred_fail_apps):

        optimal_rule = []

        for app in pred_fail_apps:

            min_loss = None
            optimal_rule.append(None)

            for rule_id, estimated_rule in enumerate(self.Estimated_rules):

                # 分析に前処理が必要であれば実行する
                try:
                    # 前シーズンから過去10個分を切り出し
                    data, _ = estimated_rule["rule"].preprocessing([u[-cfg.REVEAL_TREND-2] for u in app.featureVector[:]])
                except:
                    data = app.featureVector[-cfg.REVEAL_TREND-2]
                    print("except is called in counseling()")

                evaluate = abs(estimated_rule["rule"].dopredict(data) - app.trend[-2])[0][0]

                # 最小ロスをそのアプリの最適ルールとして保存
                if rule_id == 0:
                    min_loss = evaluate
                    optimal_rule[-1] = rule_id
                elif evaluate < min_loss:
                    min_loss = evaluate
                    optimal_rule[-1] = rule_id

        return optimal_rule


    # 最適ルールで各アプリを予測する
    # def predict_with_est_rule(self, pred_fail_apps, failed_idx, apps, optimal_rule):
    #
    #     for i, app in enumerate(apps):
    #
    #         if any([app is f_app for f_app in pred_fail_apps]):
    #
    #             # 分析に前処理が必要であれば実行する
    #             try:
    #                 # 前シーズンから過去10個分を切り出し
    #                 data, _ = self.Estimated_rules[optimal_rule[failed_idx.index(i)]]["rule"].preprocessing([u[-cfg.REVEAL_TREND-1] for u in app.featureVector[:]])
    #             except:
    #                 data = app.featureVector[-cfg.REVEAL_TREND-1]
    #                 print("except is called in predict_with_est_rule()")
    #
    #             # 予測失敗アプリのprediction_est_ruleに分析器の予測を保存
    #             self.prediction_est_rule[i].append(self.Estimated_rules[optimal_rule[failed_idx.index(i)]]["rule"].dopredict(data)[0][0])
    #
    #         # 予測が成功したアプリは分類器の予測結果を保存
    #         else:
    #             self.prediction_est_rule[i].append(self.prediction[i][-1])

    # *****************実験用の変更******************(戻さないと遅い)
    def predict_with_est_rule(self, pred_fail_apps, failed_idx, apps, optimal_rule):

        for i, app in enumerate(apps):

            data = None

            # 分析に前処理が必要であれば実行する
            try:
                # 前シーズンから過去10個分を切り出し
                data, _ = self.Estimated_rules[optimal_rule[i]]["rule"].preprocessing(
                    [u[-cfg.REVEAL_TREND-1] for u in app.featureVector[:]])
            except:
                print("self.Estimated_rules")
                print(self.Estimated_rules)
                print("optimal :")
                print(optimal_rule)
                print("failed_idx")
                print(failed_idx)
                print("failed_idx.index(i)")
                print(failed_idx.index(i))
            #     data = app.featureVector[-cfg.REVEAL_TREND-1]
            #     print("except is called in predict_with_est_rule()")

            self.prediction_only_ci[i].append(self.Estimated_rules[optimal_rule[i]]["rule"].dopredict(data)[0][0])

            if any([app is f_app for f_app in pred_fail_apps]):

                # 予測失敗アプリのprediction_est_ruleに分析器の予測を保存
                self.prediction_est_rule[i].append(self.Estimated_rules[optimal_rule[i]]["rule"].dopredict(data)[0][0])

            # 予測が成功したアプリは分類器の予測結果を保存
            else:
                self.prediction_est_rule[i].append(self.prediction[i][-1])


    # 共通ルールをマージする
    def merge_rule(self, pred_fail_apps, optimal_rules):

        rule_for_these_apps = [[] for i in range(len(self.Estimated_rules))]

        merge_num = 0

        for rule_id, estimated_rule in enumerate(self.Estimated_rules):

            for i, app in enumerate(pred_fail_apps):

                # 分析に前処理が必要であれば実行する
                try:
                    # 前シーズンから過去10個分を切り出し
                    data, _ = estimated_rule["rule"].preprocessing([u[-cfg.REVEAL_TREND-2] for u in app.featureVector[:]])
                except:
                    data = app.featureVector[-cfg.REVEAL_TREND-2]
                    print("except is called in check_existing_rule()")

                # 既存のルールを前シーズンのデータで予測して評価する
                evaluate = abs(estimated_rule["rule"].dopredict(data) - app.trend[-2])[0][0]

                # このルールによるロスが閾値未満のアプリ
                if evaluate < cfg.EVALUATE_THRESHOLD_MERGE_RULE:
                    rule_for_these_apps[rule_id].append(i)

            # いずれかのアプリの最適ルールでない場合
            if not any([rule_id is opt_rule for opt_rule in optimal_rules]):

                # このルールによるロスが閾値未満のアプリが他のルールの組み合わせと一致している場合
                if any([rule_for_these_apps[rule_id] == rule_for_these_apps[i] for i in range(len(self.Estimated_rules)) if not i == rule_id]):

                    # 削除
                    self.Estimated_rules.remove(estimated_rule)
                    merge_num += 1

        self.merge_rule_num.append(merge_num)
        if cfg.SHOW_MODEL_DETAIL and merge_num is not 0:
            print("マージされたルール数：" + str(merge_num))







if __name__ == '__main__':

    # 分類器を生成
    classifier = classify.LSTMa(cfg.TYPE_LENGTH, cfg.LSTM_REFERENCE_STEPS,
                                epochs=cfg.LSTM_EPOCHS, reference_offset=cfg.REVEAL_TREND)

    # CIMを生成
    CIM = CImodel(cfg, classifier)

    # データを生成
    data = DataStore(cfg)

    start = time.time()

    # メインループ
    for season in range(cfg.SPAN):

        if cfg.SHOW_MODEL_DETAIL:
            print("season:" + str(season))

        # 今期のデータをCIMに入力する
        new_trendrule = CIM.run(data.apps, season)

        # アプリの時系列データを更新
        for app in data.apps:
            app.update(data.trend_rule)

        # トレンドルールを更新
        data.trend_rule.update(season)
        if cfg.SHOW_MODEL_DETAIL:
            print("実存するルール数")

        if cfg.SHOW_MODEL_DETAIL:
            print("")

    end = time.time()

    print("合計時間：" + str(end - start))

    # モデルの結果を出力
    fg = figure("result/Research/research", 200, cfg.SPAN, data, CIM)

    start_offset = cfg.LSTM_REFERENCE_STEPS + cfg.REVEAL_TREND
    fg.savefig_result("PredictTrend", start_offset=start_offset)
    fg.savefig_ruleweight("TrendRuleW")
    fg.savefig_chosenrule("ChosenRule")
    fg.savefig_compare_prediction("ComparePrediction", start_offset=start_offset)
    fg.savefig_compare_prediction_ave("ComparePredictionAverage", start_offset=start_offset)
    fg.savefig_rule_num("RuleMoving", start_offset=start_offset)
    fg.save_config("config", cfg)
