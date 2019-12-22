import os
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np


class figure:


    def __init__(self, dire, dpi, span, data, CIM, learn_loss=None, eval_loss=None):

        self.dire = self.new_directory(dire)
        self.dpi = dpi

        self.span = span
        self.app = data.apps
        self.trend_rule = data.trend_rule
        self.prediction = CIM.prediction
        self.prediction_e = CIM.prediction_est_rule

        self.learn_loss = learn_loss
        self.eval_loss = eval_loss


    def new_directory(self, path):
        n = 1
        while True:
            if not os.path.exists(path + "_" + str(n)):
                os.mkdir(path + "_" + str(n))
                break
            else:
                n += 1
        return path + "_" + str(n) + "/"


    def savefig_result(self, name, start_offset=0):

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

        for i, app in enumerate(self.app):
            if self.learn_loss is not None:
                plt.scatter(x[start_offset+1:], self.learn_loss[i], color=cycle_app[i], alpha=0.3, label="learn loss (app:" + str(i) + ")")
            if self.eval_loss is not None:
                plt.scatter(x[start_offset+1:], self.eval_loss[i], color=cycle_app[i], alpha=0.3, marker="X", label="evalu loss (app:" + str(i) + ")")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.8)
        plt.savefig(self.dire + name + ".png", dpi=self.dpi)
        plt.clf()

        return


    def savefig_ruleweight(self, name):

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
        plt.savefig(self.dire + name + ".png", dpi=self.dpi)

        plt.clf()

        return


    def savefig_chosenrule(self, name):

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
        plt.savefig(self.dire + name + ".png", dpi=self.dpi)

        plt.clf()

        return


    def savefig_compare_prediction(self, name, start_offset=0):

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

            plt.plot(x[start_offset:], np.abs(np.array(self.prediction[id]) - np.array(self.app[id].trend[start_offset:])),
                     color=cycle_app[id], label="classify loss (app:" + str(id) + ")", linestyle="dotted")
            plt.plot(x[start_offset + 1:], np.abs(np.array(self.prediction_e[id]) - np.array(self.app[id].trend[start_offset + 1:])),
                     color=cycle_app[id], label="analyse loss (app:" + str(id) + ")")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.8)
        plt.savefig(self.dire + name + ".png", dpi=self.dpi)

        plt.clf()

        return
