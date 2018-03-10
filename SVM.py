from numpy import *
import matplotlib.pyplot as plt


def loaddataset(filename):
    datamat = []
    labelmat = []
    with open(filename, "r") as fr:
        for line in fr.readlines():
            linearr = line.strip().split("\t")
            datamat.append(list(map(float, linearr[0:2])))
            labelmat.append(float(linearr[2]))
    return array(datamat), array(labelmat)


def selectJrand(i, m):
    j=i
    while j==i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smosimple(datamat, classlabels, C, toler, maxIter):
    datamatrix = mat(datamat)
    labelmat = mat(classlabels).T
    b = 0
    m, n = shape(datamatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(multiply(alphas, labelmat).T*(datamatrix*datamatrix[i, :].T)) + b
            Ei = fxi - float(labelmat[i])
            if (labelmat[i] * Ei < -toler and alphas[i] < C) or (labelmat[i] * Ei > toler and alphas[i] > 0):
                j = selectJrand(i, m)
                fxj = float(multiply(alphas, labelmat).T*(datamatrix*datamatrix[j, :].T)) + b
                Ej = fxj - float(labelmat[j])
                alphaiold = alphas[i].copy()
                alphajold = alphas[j].copy()
                if labelmat[i] != labelmat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[j])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = datamatrix[i, :] * datamatrix[i, :].T + datamatrix[j, :] * datamatrix[j, :].T - 2.0*datamatrix[i, :] * datamatrix[j, :].T
                alphas[j] += labelmat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphajold) < 0.00001:
                    print("alpha_j not moving enough")
                    continue
                alphas[i] += labelmat[i]*labelmat[j]*(alphajold - alphas[j])
                b1 = b - Ei - labelmat[i] * (alphas[i] - alphaiold)*datamatrix[i, :] * datamatrix[i, :].T-\
                    labelmat[j] * (alphas[j] - alphajold)* \
                    datamatrix[i, :] * datamatrix[j, :].T
                b2 = b - Ej - labelmat[i] * (alphas[i] - alphaiold)*datamatrix[i, :] * datamatrix[j, :].T-\
                    labelmat[j] * (alphas[j] - alphajold)* \
                    datamatrix[j, :] * datamatrix[j, :].T
                if alphas[i] >0 and alphas[i] < C:
                    b = b1
                elif alphas[j] >0 and alphas[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2
                alphaPairsChanged +=1
                print("iter: %d, i:%d, pairs changed %d" %(iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter+=1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas
if __name__ == "__main__":
    data, label = loaddataset("testSet.txt")
    b, alphas = smosimple(data, label, 0.6, 0.001, 40)
    w = multiply(alphas, mat(label).T).T*mat(data)
    y = w * mat(data).T + b
    z = array(y)[0]
    for i in range(z.shape[0]):
        if z[i] > 0:
            z[i] = 1
        else:
            z[i] = -1
    print(sum(z==label))
