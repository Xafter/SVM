from svm_complete import *


def img2vector(dirName):
    from os import listdir
    digitlist = listdir(dirName)
    datamat = []

    for each in digitlist:
        with open(dirName+"\\"+each, 'r') as f:
            datamat.append(list(map(float,list(''.join(f.read().split('\n'))))))
    datamat = array(datamat)
    return datamat


def loadImages(dirName):
    from os import listdir
    digitlist = listdir(dirName)
    hwlabel = []
    m = len(digitlist)
    for i in range(m):
        label = int(digitlist[i].split('.')[0].split('_')[0])
        if label == 1:
            hwlabel.append(1)
        else:
            hwlabel.append(-1)
    trainmat = img2vector(dirName)
    return trainmat, array(hwlabel)


def test(kTup=('rbf', 5)):
    data, label = loadImages("trainingDigits")
    b, alphas = smoP(data, label, 200, 0.0001, 10000, kTup)
    test, testlabel = loadImages("testDigits")
    testmat = mat(test)
    m, n = test.shape
    count = 0
    pre = []
    for i in range(m):
        predicts = predict(b, alphas, kTup, data, label, testmat[i])
        pre.append(predicts)
        if predicts == testlabel[i]:
            count += 1
    y = vstack((pre, testlabel))
    print(count/m)
test()