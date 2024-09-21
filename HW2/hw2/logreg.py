import numpy as np
import argparse
import numpy.random as nr
import sys
import pdb

class LogReg(object):
    # W1: d x d, W2: d x 1, b: scalar
    def __init__(self, d, lamb):
        self.W1 = nr.randn(d, d) # Create d x d array (matrix) with random number
        self.W2 = nr.randn(d) #  Create d length of array with random number
        self.b = nr.randn() # it's just scalar with random number
        self.lamb = lamb # To be multiplied on the regularization term.

    def foo(self, x):
        return np.dot(x, np.dot(self.W1, x)) + np.dot(x, self.W2) + self.b

    def logistic(self, x):
        return 1 / (1 + np.exp(-self.foo(x)))

    # TODO: Implement the log-likelihood function of the data
    def log_likelihood(self, data, labels):
        # Initialize the list label = 1 or not
        list_label_true = []
        list_label_false = []

        # Fill the datas according to the labels
        for i in range(len(labels)):
            if labels[i] == 1:
                list_label_true.append(data[i])
            else:
                list_label_false.append(data[i])

        # logL(θ;D)= ∑ logσ(x_i) + ∑ logσ(1 - x_i)
        log_likelihood_value = (
                # here for the label = 1
                np.sum(np.log(self.logistic(list_label_true))) +
                # here for the label = 1, so add -1
                np.sum(np.log(1 - self.logistic(list_label_false)))
        )

        return log_likelihood_value

    # TODO: Implement the derivatives of the log-likelihood w.r.t. each parameter, evaluated at x (x is d x 1)
    def dLLdW1(self, x):

        sigma_z = self.logistic(x)

        for i in range(self.W1.shape[0]):
            y =

        pass
  
    def dLLdW2(self, x):
        pass

    def dLLdb(self, x):
        pass

    # TODO: Return the L2 regularization term.
    def l2_reg(self):
        pass

    # TODO: Implement the classification rule on x (i.e., what is the label of x?)
    def predict(self, x):
        pass

    # TODO: Implement a single gradient ascent step, and return the objective (*not* the log-likelihood!!)
    # Question: What is the objective?
    def step(self, dat, lbl, lr):
        pass

    # TODO: Implement the F1 measure computation. 
    def f1(step, test_data, test_gt):
        pass


# Returns a numpy array of size n x d, where n is the number of samples and d is the dimensions
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
    num_epoch = args.epochs # How many times u want to train?
    learning_rate = args.lr # step for learning
    lamb = args.lambdaValue  # To be multiplied on the regularization term.

    tr_data, tr_gt = read_data('train.csv') # originally train.csv has 1000 x 11 matrix, with label for the first column
    va_data, va_gt = read_data('val.csv')
    te_data, te_gt = read_data('test.csv')

    print("training data shape [-1]: ", tr_data.shape[-1]) # get column number for training data set = 10
    # So training set has 1000 data with 10-dimension vector

    # W1: 10x10, W2: 10x1, b: scalar, lambda: 1e-2 (default)
    model = LogReg(tr_data.shape[-1], lamb)

    # print(tr_data.shape) # 1000 x 10 shape training data set
    # print(len(tr_gt)) # label for the 1000 training set
    #
    # print(va_data.shape) # 100 x 10 shape validate data set
    # print(len(va_gt)) # label for the 100 validate set
    #
    # print(te_data.shape) # 100 x 10 shape test data set
    # print(len(te_gt)) # label for the 1000 test set


    # An EPOCH is a single pass over the entire dataset.
    # Normally, we'd run this epoch loop until the learning has converged, but we'll
    # just run a fixed number of loops for this assignment.

    for ep in range(num_epoch):
        loss = model.step(tr_data, tr_gt, learning_rate)
        # Maybe add your own learning rate scheduler here?
        print('[Epoch {}] Regularized loss = {}'.format(ep, loss))

    print('F1 score on test data = {}'.format(model.f1(te_data, te_gt)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambdaValue", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    main(args)
