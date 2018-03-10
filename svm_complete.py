from numpy import *


def loaddataset(filename):
    datamat = []
    labelmat = []
    with open(filename, "r") as fr:
        for line in fr.readlines():
            linearr = line.strip().split("\t")
            datamat.append(list(map(float, linearr[0:2])))
            labelmat.append(float(linearr[2]))
    return array(datamat), array(labelmat)


# 核函数
def kerneltrans(x, y, kTup):
    if kTup[0]  == 'lin':
        K = x * y.T
    elif kTup[0] == 'rbf':
        delta = (x-y) * (x-y).T
        K = exp(delta/(-2*kTup[1]**2))
    else:
        raise NameError("houston we have a problem--that kernel is not recognized")
    return K



class Optstruct:
    def __init__(self, datamat, classlabels, C, toler, kTup):
        self.X = datamat
        self.labelmat = classlabels
        self.C = C
        self.tol = toler
        self.m = datamat.shape[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            for j in range(self.m):
                self.K[i, j] = kerneltrans(self.X[i, :], self.X[j, :], kTup)


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


def calcEk(os, k):
    fxk = float(multiply(os.alphas, os.labelmat).T * os.K[:, k] + os.b)
    Ek = fxk - float(os.labelmat[k])
    return Ek


def selectJ(i, os, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    os.eCache[i] = [1, Ei]
    validEcachearray = nonzero(os.eCache[:, 0].A)[0]
    if len(validEcachearray) > 1:
        for k in validEcachearray:
            Ek = calcEk(os, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, os.m)
        Ej = calcEk(os, j)
        return j, Ej


def updateEk(os, k):
    Ek = calcEk(os, k)
    os.eCache[k] = [1, Ek]


def innerL(i, os):
    Ei = calcEk(os, i)
    if (os.labelmat[i] * Ei < -os.tol and os.alphas[i] < os.C) or (os.labelmat[i] * Ei > os.tol and os.alphas[i] > 0):
        j, Ej = selectJ(i, os, Ei)
        alphaiold = os.alphas[i].copy()
        alphajold = os.alphas[j].copy()
        if os.labelmat[i] != os.labelmat[j]:
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if  L == H:
            print("L == H")
            return 0
        eta = 2.0 * os.K[i, j] - os.K[i, i] - os.K[j, j]
        if eta >= 0 :
            print("eta>=0")
            return 0
        os.alphas[j] -= os.labelmat[j]*(Ei - Ej)/eta
        os.alphas[j] = clipAlpha(os.alphas[j], H, L)
        updateEk(os, j)
        if abs(os.alphas[j] - alphajold) < 0.00001:
            print("j not moving")
            return 0
        os.alphas[i] += os.labelmat[j]*os.labelmat[i]*(alphajold - os.alphas[j])
        updateEk(os, i)
        b1 = os.b - Ei - os.labelmat[i] * (os.alphas[i] - alphaiold)*os.K[i ,i] - os.labelmat[j]*\
            (os.alphas[j] - alphajold)*os.K[i, j]
        b2 = os.b - Ej - os.labelmat[i] * (os.alphas[i] - alphaiold)*os.K[i, j] - os.labelmat[j]*\
            (os.alphas[j] - alphajold)*os.K[j, j]
        if os.alphas[i] > 0 and os.alphas[i] < os.C:
            os.b = b1
        elif os.alphas[j] > 0 and os.alphas[j] < os.C:
            os.b = b2
        else:
            os.b = (b1+b2)/2
        return 1
    else:
        return 0


def smoP(datamat, classlabels, C, tolers, maxiter, kTup = ('lin', 0)):
    os = Optstruct(mat(datamat), mat(classlabels).T, C, tolers, kTup)
    iter = 0
    entireset = True
    alphaPairsChanged = 0
    while iter < maxiter and (alphaPairsChanged > 0 or entireset):
        if entireset:
            for i in range(os.m):
                alphaPairsChanged += innerL(i, os)
                print("entireset iter: %d, i: %d, pairs changed %d" %(iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonbound = nonzero((os.alphas.A > 0) * (os.alphas.A < 0))[0]
            for i in nonbound:
                alphaPairsChanged += innerL(i, os)
                print("nonbound iter: %d, i: %d, pairs changed %d" %(iter, i, alphaPairsChanged))
            iter += 1
        if entireset:
            entireset = False
        elif alphaPairsChanged == 0:
            entireset = True
        print("iteration number: %d" % iter)
    return os.b, os.alphas


def calcW(alphas, data, classlabels):
    alpha = mat(alphas).T
    label = mat(classlabels)
    data = mat(data)
    w = multiply(alpha, label) * data
    return w


def classify(w, point, b):
    fx = w * point.T + b
    return fx


def predict(b, alphas, kTup, data, label, point):
    datamat = mat(data)
    labelmat = mat(label).T
    svInd = nonzero(alphas.A>0)[0]
    svs = datamat[svInd]
    labelsv = labelmat[svInd]
    m, n = shape(svs)
    predicts = 0
    for i in range(m):
        kernel = kerneltrans(svs[i], point, kTup)
        predicts += kernel * multiply(labelsv[i], alphas[svInd][i])
    return float(sign(predicts))


def testrbf(k1 = 1.3):
    data, label = loaddataset("testSetRBF.txt")
    kTup = ('rbf', k1)
    b, alphas = smoP(data[0:80], label[0:80], 0.6, 0.001, 40, kTup)
    testmat = mat(data[80:])
    m, n = testmat.shape
    count = 0
    pre = []
    for i in range(m):
        predicts = predict(b, alphas, kTup, data[0:80], label[0:80], testmat[i])
        pre.append(predicts)
        if predicts == label[80:][i]:
            count += 1
    y = vstack((pre, label[80:]))
    print(count/m)




if __name__ == '__main__':
    testrbf(k1=0.03)
