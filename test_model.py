import sampledata, classify

from matplotlib import pyplot as plt
import numpy as np
import os


'''
        モデルを実行する関数を置く場所
'''


def save_fig(path, loss=False):
    n = 1
    if(loss == False):
        while(True):
            if(os.path.exists("result/" + path + "/predict" + str(sampledata.OneDimTimeSeries.noise_type)
                              +"_"+str(n) + ".png") == False):
                break
            else:
                n += 1
        plt.savefig("result/" + path + "/predict" + str(sampledata.OneDimTimeSeries.noise_type)
                              +"_"+str(n)+".png")
    else:
        while (True):
            if (os.path.exists("result/" + path + "/loss" + str(sampledata.OneDimTimeSeries.noise_type)
                               + "_" + str(n) + ".png") == False):
                break
            else:
                n += 1
        plt.savefig("result/" + path + "/loss" + str(sampledata.OneDimTimeSeries.noise_type)
                    + "_" + str(n) + ".png")



def run_OCsvm():
    model = classify.OneClassSVM

    for i in range(1):
        data, ano_data, label = sampledata.OneDimTimeSeries.make_value()
        dataLen = sampledata.OneDimTimeSeries.endTIME + 1

        a = [0] * dataLen
        data = np.array(data + a)
        data = data.reshape(2, dataLen).T

        model.dofit(data, label)

    pred = model.dopredict(data)

    #結果を出力しやすい形に変える
    for i in range(dataLen):
        if (pred[i] == 1):
            pred[i] = -15
        else:
            pred[i] = data[i][0]

    x = list(range(len(ano_data)))
    plt.plot(x, data)
    plt.scatter(x, pred, marker="o", color="green")
    plt.scatter(x, ano_data, marker=".", color="red")
    save_fig("OCsvm")
    plt.show()



def run_knn(Threshold = 0.01):
    model = classify.kNN


    data, ano_data, _ = sampledata.OneDimTimeSeries.make_value()
    dataLen = sampledata.OneDimTimeSeries.endTIME + 1


    #sklearnは2次元行列に変換しないと扱えない
    a = [0] * dataLen
    data = np.array(data + a)
    data = data.reshape(2, dataLen).T

    model.dofit(data)
    dist = model.dopredict(data)

    pred = []

    # plt.plot(dist)
    # plt.show()

    #結果を出力しやすい形に変える
    for i in range(dataLen):
        if (dist[i][1] < Threshold):
            pred.append(-15)
        else:
            pred.append(data[i][0])

    x = list(range(len(ano_data)))
    plt.plot(x, data)
    plt.scatter(x, pred, marker="o", color="green")
    plt.scatter(x, ano_data, marker=".", color="red")
    save_fig("knn")
    plt.show()



def run_LPF(Threshold):
    data, ano_data, _ = sampledata.OneDimTimeSeries.make_value()
    lowpassed = classify.LPF.model(data)
    dataLen = sampledata.OneDimTimeSeries.endTIME + 1

    pred = []
    thLPF_o = []
    thLPF_u = []

    for i in range(dataLen):
        if (abs(lowpassed[i] - data[i]) < Threshold):
            pred.append(-15)
        else:
            pred.append(data[i])

        thLPF_o.append(lowpassed[i] + Threshold)
        thLPF_u.append(lowpassed[i] - Threshold)


    x = list(range(len(ano_data)))
    plt.plot(x, data)
    plt.plot(x, lowpassed)
    # plt.plot(x, thLPF_o, linestyle="dashed")
    # plt.plot(x, thLPF_u, linestyle="dashed")
    plt.scatter(x, pred, marker="o", color="green")
    plt.scatter(x, ano_data, marker=".", color="red")
    save_fig("LPF")
    plt.show()



def run_LSTM(Threshold):
    raw_data, _, label = sampledata.OneDimTimeSeries.make_value()
    data, target, label = classify.LSTMs.preprocessing(np.array(raw_data), np.array(label))

    model = classify.LSTMs.modeling()
    history, predicted = classify.LSTMs.dofit(data, target, model, label)
    # future_result = classify.LSTMs.predict_future(data, model)

    length = classify.LSTMs.reference_len

    pred = []
    for i in range(len(predicted)):
        #labelを予測
        if (predicted[i] > Threshold):
            pred.append(-15)
        else:
            pred.append(raw_data[i+length])

        # #targetを予測
        # if (abs(predicted[i]*13 - raw_data[i+12]) < Threshold):
        #     pred.append(-15)
        # else:
        #     pred.append(raw_data[i+12])

    # Plot Wave
    plt.figure()
    plt.plot(range(0, len(raw_data)), raw_data, color="b", label="data")
    plt.plot(range(length, len(predicted) + length), predicted, color="r", label="rawOutput")
    plt.scatter(range(length, len(pred) + length), pred, color="g", label="RecogAnomaly")
    plt.scatter(range(length, len(label) + length), label, color="y", label="TruthAnomaly")
    # plt.plot(range(0 + len(raw_data) - 24, len(future_result) + len(raw_data) - 24), future_result, color="orange",
    #          label="future")
    plt.legend()
    save_fig("lstm")
    plt.show()
    plt.close()

    # Plot Training loss & Validation Loss
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()
    save_fig("lstm", loss=True)
    plt.close()



def run_AutoEncoder(Threshold):
    raw_data, _, label = sampledata.OneDimTimeSeries.make_value()
    data, target = classify.AutoEncoder.preprocessing(np.array(raw_data))

    model = classify.AutoEncoder.modeling()
    history, predicted = classify.AutoEncoder.dofit(data, model)

    length = classify.AutoEncoder.reference_len

    pred = []
    for i in range(len(predicted)):
        # #labelを予測
        # if (predicted[i] > 0):
        #     pred.append(-15)
        # else:
        #     pred.append(raw_data[i+length])

        #targetを予測
        if (abs(predicted[i]*13 - raw_data[i+length]) < Threshold):
            pred.append(-15)
        else:
            pred.append(raw_data[i+length])

    # Plot Wave
    plt.figure()
    plt.plot(range(0, len(raw_data)), raw_data, color="b", label="raw")
    plt.plot(range(length, len(predicted) + length), predicted*13, color="r", label="predict")
    plt.scatter(range(length, len(pred) + length), pred, color="g", label="anomaly")
    # plt.plot(range(0 + len(raw_data) - 24, len(future_result) + len(raw_data) - 24), future_result, color="orange",
    #          label="future")
    plt.legend()
    save_fig("AutoEncoder")
    plt.show()
    plt.close()

    # Plot Training loss & Validation Loss
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()
    save_fig("AutoEncoder", loss=True)
    plt.close()


if __name__ == '__main__':
    #run_knn(0.002)
    #run_OCsvm()
    #run_LPF(1)
    run_LSTM(0.5)
    #run_AutoEncoder(1)
