import numpy as np
import argparse
import numpy.random as nr
import sys
import pdb

class LogReg(object):
    def __init__(self, d, lamb):
        self.W1 = nr.randn(d, d)
        self.W2 = nr.randn(d)
        self.b = nr.randn()
        self.lamb = lamb

    def foo(self, x):
        return np.dot(x, np.dot(self.W1, x)) + np.dot(x, self.W2) + self.b

    def logistic(self, x):
        return 1 / (1 + np.exp(-self.foo(x)))

    def log_likelihood(self, data, labels):
        ll = 0
        for x, y in zip(data, labels):
            p = self.logistic(x)
            ll += y * np.log(p) + (1 - y) * np.log(1 - p)
        return ll

    def dLLdW1(self, x):
        p = self.logistic(x)
        return np.outer(x, x) * (1 - p)

    def dLLdW2(self, x):
        p = self.logistic(x)
        return x * (1 - p)

    def dLLdb(self, x):
        p = self.logistic(x)
        return 1 - p

    def l2_reg(self):
        return 0.5 * self.lamb * (np.sum(self.W1**2) + np.sum(self.W2**2) + self.b**2)

    def predict(self, x):
        return 1 if self.logistic(x) > 0.5 else 0

    def step(self, dat, lbl, lr):
        for x, y in zip(dat, lbl):
            self.W1 += lr * (self.dLLdW1(x) * y - self.lamb * self.W1)
            self.W2 += lr * (self.dLLdW2(x) * y - self.lamb * self.W2)
            self.b += lr * (self.dLLdb(x) * y - self.lamb * self.b)
        return -self.log_likelihood(dat, lbl) + self.l2_reg()

    def f1(self, test_data, test_gt):
        predictions = [self.predict(x) for x in test_data]
        tp = sum([1 for p, t in zip(predictions, test_gt) if p == 1 and t == 1])
        fp = sum([1 for p, t in zip(predictions, test_gt) if p == 1 and t == 0])
        fn = sum([1 for p, t in zip(predictions, test_gt) if p == 0 and t == 1])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def read_data(filename):
    data = []
    labels = []
    with open(filename, 'r') as f:
        tmp = f.readlines()
    for line in tmp:
        toks = line.split(',')
        lbl = int(toks[0])
        dat = np.array([float(x.strip()) for x in toks[1:]])
        if len(data) == 0: data = dat
        else: data = np.vstack((data, dat))
        labels.append(lbl)
    return data, labels

def main(args): 
    num_epoch = args.epochs
    learning_rate = args.lr
    lamb = args.lamb

    tr_data, tr_gt = read_data('train.csv')
    va_data, va_gt = read_data('val.csv')
    te_data, te_gt = read_data('test.csv')
    model = LogReg(tr_data.shape[-1], lamb)

    for ep in range(num_epoch):
        loss = model.step(tr_data, tr_gt, learning_rate)
        print('[Epoch {}] Regularized loss = {}'.format(ep, loss))

    print('F1 score on test data = {}'.format(model.f1(te_data, te_gt)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lamb", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    main(args)
