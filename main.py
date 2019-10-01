import sampledata, augment, classify, analysis, Flist



def CImodel():

    season = 0

    while True:

        data = sampledata.OneDimTimeSeries.make_value()

        classed = classify.modelC.testF(data)

        augmented = augment.fooAugment()

        analysed = analysis.modelA.testF()

        Flist.refreshList()

        season += 1


if __name__ == '__main__':
    CImodel()