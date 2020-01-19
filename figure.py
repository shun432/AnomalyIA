import os
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np


class figure:


    def __init__(self, dire, dpi, span, data, CIM, learn_loss=None, eval_loss=None, different_dir_app=True):

        self.dire = self.new_num_directory(dire)
        self.app_dire = [self.make_num_directory("app", i) for i in range(data.app_num)]
        self.trend_dire = [self.make_num_directory("trend", i) for i in range(len(data.trend_rule.w))]
        self.dpi = dpi

        self.span = span
        self.app = data.apps
        self.trend_rule = data.trend_rule
        self.prediction = CIM.prediction
        self.prediction_e = CIM.prediction_est_rule

        self.predfail_app_num = CIM.predfail_app_num
        self.rule_num = CIM.rule_num
        self.add_rule_num = CIM.add_rule_num
        self.lost_rule_num = CIM.lost_rule_num
        self.useless_rule_num = CIM.useless_rule_num
        self.merge_rule_num = CIM.merge_rule_num

        self.learn_loss = learn_loss
        self.eval_loss = eval_loss

        self.diff_dir = different_dir_app


    def new_num_directory(self, path):
        n = 1
        while True:
            if not os.path.exists(path + "_" + str(n)):
                os.mkdir(path + "_" + str(n))
                break
            else:
                n += 1
        return path + "_" + str(n) + "/"


    def make_num_directory(self, name, num):

        os.mkdir(self.dire + "/" + name + "_" + str(num))

        return self.dire + "/" + name + "_" + str(num) + "/"


    def find_min_max(self, data_list, length, standarize_zero=True):

        if standarize_zero:
            min = 0
            max = 0
        else:
            min = data_list[0][0]
            max = data_list[0][0]

        for data in data_list:

            for j in range(length):

                if j < len(data):
                    if data[j] < min:
                        min = data[j]
                    if data[j] > max:
                        max = data[j]

        return min, max


    def savefig_result(self, name, start_offset=0):

        x = list(range(self.span))

        if self.diff_dir:

            # トレンドルールごとの色（chosenRuleより）
            if len(self.trend_rule.w) <= 10:
                cycle_tr = plt.rcParams['axes.prop_cycle'].by_key()['color']
            elif len(self.trend_rule.w) <= 20:
                cycle_tr = plt.cm.get_cmap('tab20').colors
            else:
                cycle_tr = list(colors.XKCD_COLORS.items())[:100]

            for i, app in enumerate(self.app):

                min, max = self.find_min_max([self.prediction[i], self.prediction_e[i]], self.span)

                plt.figure(figsize=(len(x) / 10, 5.5))

                # （chosenRuleより）
                for j in range(len(self.trend_rule.w)):
                    plt.fill_between([j - 0.5, j + 0.5], [max * 1.1 + 0.1, max * 1.1 + 0.1],
                                     [min * 1.1 - 0.1, min * 1.1 - 0.1],
                                     facecolor=cycle_tr[j], alpha=0.2,
                                     label="Chosenrule:" + str(j))
                for j in range(self.span):
                    plt.fill_between([j - 0.5, j + 0.5], [max*1.1+0.1, max*1.1+0.1], [min*1.1-0.1, min*1.1-0.1],
                                     facecolor=cycle_tr[self.app[i].trend_idx[j]], alpha=0.2)


                plt.plot(x, app.trend, label="trend", linestyle="dotted", color="black")
                plt.plot(x[start_offset:], self.prediction[i], label="classifier pred", color="blue")
                plt.plot(x[start_offset + 1:], self.prediction_e[i], label="analyser pred", color="orange")

                if self.learn_loss is not None:
                    plt.scatter(x[start_offset + 1:], self.learn_loss[i], alpha=0.3,
                                label="learn loss")
                if self.eval_loss is not None:
                    plt.scatter(x[start_offset + 1:], self.eval_loss[i], alpha=0.3, marker="X",
                                label="eval loss")

                plt.xlabel('season')
                plt.ylabel('trend value')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.subplots_adjust(right=0.8)
                plt.savefig(self.app_dire[i] + name + ".png", dpi=self.dpi)
                plt.clf()

        else:

            plt.figure(figsize=(len(x)/10, 5.5))

            # アプリごとの色
            if len(self.app) <= 10:
                cycle_app = plt.rcParams['axes.prop_cycle'].by_key()['color']
            elif len(self.app) <= 20:
                cycle_app = plt.cm.get_cmap('tab20').colors
            else:
                cycle_app = list(colors.XKCD_COLORS.items())[:100]

            for i, app in enumerate(self.app):
                plt.plot(x, self.app[i].trend, color=cycle_app[i], label="trend (app:" + str(i) + ")", linestyle="dotted")
                plt.plot(x[start_offset:], self.prediction[i], color=cycle_app[i], label="pred (app:" + str(i) + ")")

                if self.learn_loss is not None:
                    plt.scatter(x[start_offset+1:], self.learn_loss[i], color=cycle_app[i], alpha=0.3,
                                label="learn loss (app:" + str(i) + ")")
                if self.eval_loss is not None:
                    plt.scatter(x[start_offset+1:], self.eval_loss[i], color=cycle_app[i], alpha=0.3, marker="X",
                                label="evalu loss (app:" + str(i) + ")")

            plt.xlabel('season')
            plt.ylabel('trend value')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.8)
            plt.savefig(self.dire + name + ".png", dpi=self.dpi)
            plt.clf()

        return


    def savefig_ruleweight(self, name):

        x = list(range(self.span))

        if self.diff_dir:

            # 特徴ごとの色
            if len(self.trend_rule.w[0]) <= 10:
                cycle_ft = plt.rcParams['axes.prop_cycle'].by_key()['color']
            elif len(self.trend_rule.w[0]) <= 20:
                cycle_ft = plt.cm.get_cmap('tab20').colors
            else:
                cycle_ft = list(colors.XKCD_COLORS.items())[:100]

            for i in range(len(self.trend_rule.w)):

                plt.figure(figsize=(len(x) / 10, 5.5))

                # 特徴毎に
                for j in range(len(self.trend_rule.w[i])):
                    plt.plot(x, self.trend_rule.w[i][j][:-1], color=cycle_ft[j], label="feature:" + str(j))

                plt.xlabel('season')
                plt.ylabel('weight of trend rule')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.subplots_adjust(right=0.8)
                plt.savefig(self.trend_dire[i] + name + ".png", dpi=self.dpi)

                plt.clf()


        else:

            plt.figure(figsize=(len(x)/10, 5.5))

            # トレンドルールごとの色
            if len(self.trend_rule.w) <= 10:
                cycle_tr = plt.rcParams['axes.prop_cycle'].by_key()['color']
            elif len(self.trend_rule.w) <= 20:
                cycle_tr = plt.cm.get_cmap('tab20').colors
            else:
                cycle_tr = list(colors.XKCD_COLORS.items())[:100]

            # 特徴ごとの色
            if len(self.trend_rule.w[0]) <= 10:
                cycle_ft = plt.rcParams['axes.prop_cycle'].by_key()['color']
            elif len(self.trend_rule.w[0]) <= 20:
                cycle_ft = plt.cm.get_cmap('tab20').colors
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

            plt.xlabel('season')
            plt.ylabel('weight of trend rule')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.8)
            plt.savefig(self.dire + name + ".png", dpi=self.dpi)

            plt.clf()

        return


    def savefig_chosenrule(self, name):

        x = list(range(self.span))

        if self.diff_dir:

            pass        # savefig_resultに統合

        else:

            plt.figure(figsize=(len(x)/10, 5.5))

            # アプリごとの色
            if len(self.app) <= 10:
                cycle_app = plt.rcParams['axes.prop_cycle'].by_key()['color']
            elif len(self.app) <= 20:
                cycle_app = plt.cm.get_cmap('tab20').colors
            else:
                cycle_app = list(colors.XKCD_COLORS.items())[:100]

            # トレンドルールごとの色
            if len(self.trend_rule.w) <= 10:
                cycle_tr = plt.rcParams['axes.prop_cycle'].by_key()['color']
            elif len(self.trend_rule.w) <= 20:
                cycle_tr = plt.cm.get_cmap('tab20').colors
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

            plt.xlabel('シーズン')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.8)
            plt.savefig(self.dire + name + ".png", dpi=self.dpi)

            plt.clf()

        return


    def savefig_compare_prediction(self, name, start_offset=0):

        x = list(range(self.span))

        if self.diff_dir:

            for i in range(len(self.app)):

                plt.figure(figsize=(len(x) / 10, 5.5))

                plt.plot(x[start_offset:],
                         np.abs(np.array(self.prediction[i]) - np.array(self.app[i].trend[start_offset:])),
                         label="classify loss", linestyle="dotted")
                plt.plot(x[start_offset + 1:],
                         np.abs(np.array(self.prediction_e[i]) - np.array(self.app[i].trend[start_offset + 1:])),
                         label="analyse loss")

                plt.xlabel('season')
                plt.ylabel('prediction loss')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.subplots_adjust(right=0.8)
                plt.savefig(self.app_dire[i] + name + ".png", dpi=self.dpi)

                plt.clf()

        else:

            plt.figure(figsize=(len(x)/10, 5.5))

            # アプリごとの色
            if len(self.app) <= 10:
                cycle_app = plt.rcParams['axes.prop_cycle'].by_key()['color']
            elif len(self.app) <= 20:
                cycle_app = plt.cm.get_cmap('tab20').colors
            else:
                cycle_app = list(colors.XKCD_COLORS.items())[:100]

            for id in range(len(self.app)):

                plt.plot(x[start_offset:], np.abs(np.array(self.prediction[id]) - np.array(self.app[id].trend[start_offset:])),
                         color=cycle_app[id], label="classify loss (app:" + str(id) + ")", linestyle="dotted")
                plt.plot(x[start_offset + 1:], np.abs(np.array(self.prediction_e[id]) - np.array(self.app[id].trend[start_offset + 1:])),
                         color=cycle_app[id], label="analyse loss (app:" + str(id) + ")")

            plt.xlabel('season')
            plt.ylabel('prediction loss')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.8)
            plt.savefig(self.dire + name + ".png", dpi=self.dpi)

            plt.clf()

        return


    def savefig_compare_prediction_ave(self, name, start_offset=0):

        x = list(range(self.span))

        if self.diff_dir:

            prediction = []
            prediction_e = []

            for j in range(self.span - start_offset):

                sum = 0
                sum_e = 0

                for i in range(len(self.app)):

                    sum += abs(self.prediction[i][j] - self.app[i].trend[j + start_offset])
                    if j is not self.span - start_offset - 1:
                        sum_e += abs(self.prediction_e[i][j] - self.app[i].trend[j + start_offset + 1])

                prediction.append(sum / len(self.app))
                if j is not self.span - start_offset - 1:
                    prediction_e.append(sum_e / len(self.app))

            plt.figure(figsize=(len(x) / 10, 5.5))

            plt.xlabel('season')
            plt.ylabel('prediction loss average')
            plt.plot(x[start_offset:], prediction, label="classify loss", linestyle="dotted")
            plt.plot(x[start_offset + 1:], prediction_e, label="analyse loss")

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.8)
            plt.savefig(self.dire + name + ".png", dpi=self.dpi)

            plt.clf()


    def savefig_rule_num(self, name, start_offset=0):

        x = list(range(self.span))

        plt.figure(figsize=(len(x)/10, 5.5))

        chart_num = 6
        width = 0.8 / chart_num

        plt.plot(x[start_offset + 1:], self.predfail_app_num, label="prediction fail app")
        plt.plot(x[start_offset + 1:], self.rule_num, label="rule")
        plt.plot(x[start_offset + 1:], self.add_rule_num, label="add rule")
        plt.plot(x[start_offset + 1:], self.lost_rule_num, label="lost rule")
        plt.plot(x[start_offset + 1:], self.useless_rule_num, label="useless rule")
        plt.plot(x[start_offset + 1:], self.merge_rule_num, label="merge rule")

        plt.xlabel('season')
        plt.ylabel('number')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.8)
        plt.savefig(self.dire + name + ".png", dpi=self.dpi)

        plt.clf()

        return

    def save_config(self, name, cfg):

        import json

        setting = dict(
            APP_NUM = cfg.APP_NUM,
            SPAN = cfg.SPAN,
            SHIFT_TREND_RULE = cfg.SHIFT_TREND_RULE,
            EVALUATE_THRESHOLD_PRED_FAIL = cfg.EVALUATE_THRESHOLD_PRED_FAIL,
            SAMPLING = cfg.SAMPLING,
            EVALUATE_THRESHOLD_DELETE_RULE = cfg.EVALUATE_THRESHOLD_DELETE_RULE,
            EVALUATE_THRESHOLD_ADD_RULE = cfg.EVALUATE_THRESHOLD_ADD_RULE,
            EVALUATE_THRESHOLD_MERGE_RULE = cfg.EVALUATE_THRESHOLD_MERGE_RULE,
            THRESHOLD_APPNUM = cfg.THRESHOLD_APPNUM,
            TRY_NEWRULE_NUM = cfg.TRY_NEWRULE_NUM,
            LSTM_REFERENCE_STEPS = cfg.LSTM_REFERENCE_STEPS,
            LSTM_EPOCHS = cfg.LSTM_EPOCHS,
            NN_EPOCHS = cfg.NN_EPOCHS,
            DATATYPE = [dict(
                name = feat["name"],
                type = str(type(feat["data"]))
            ) for feat in cfg.DATATYPE],
            FIRST_BIN = cfg.FIRST_BIN,
            FIRST_W = cfg.FIRST_W
        )

        fw = open(self.dire + name + '.json', 'w')
        json.dump(setting, fw, indent=4)

        return