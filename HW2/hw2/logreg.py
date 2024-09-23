import numpy as np
import argparse
import numpy.random as nr
import sys
import pdb
import matplotlib.pyplot as plt

class LogReg(object):
    # W1: d x d, W2: d x 1, b: scalar
    def __init__(self, d, lamb):
        self.W1 = nr.randn(d, d) # Create d x d array (matrix) with random number
        self.W2 = nr.randn(d) #  Create d length of array with random number
        self.b = nr.randn() # it's just scalar with random number
        self.lamb = lamb # To be multiplied on the regularization term.

    def foo(self, x):
        # x^T*W1*x + x^T*W2 + b -> it is z
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

        # Convert lists to numpy arrays
        array_label_true = np.array(list_label_true)
        array_label_false = np.array(list_label_false)

        # Apply logistic function to each data point
        logistic_true = np.apply_along_axis(self.logistic, 1, array_label_true)
        logistic_false = np.apply_along_axis(self.logistic, 1, array_label_false)

        # logL(θ;D)= ∑ logσ(x_i) + ∑ logσ(1 - x_i)
        log_likelihood_value = (
            # here for the label = 1
                np.sum(np.log(logistic_true)) +
                # here for the label = 0
                np.sum(np.log(1 - logistic_false))
        )

        return log_likelihood_value

    # TODO: Implement the derivatives of the log-likelihood w.r.t. each parameter, evaluated at x (x is d x 1)
    def dLLdW1(self, x):
        derivative = np.outer(x, x)
        #logistic = self.logistic(x)
        return derivative

    def dLLdW2(self, x):
        derivative = x
        # logistic = self.logistic(x)
        return derivative

    def dLLdb(self, x):
        derivative = 1
        #logistic = self.logistic(x)
        return derivative

    # TODO: Return the L2 regularization term.
    def l2_reg(self):
        sum = np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.b**2)
        return self.lamb * sum

    # TODO: Implement the classification rule on x (i.e., what is the label of x?)
    def predict(self, x):
        if self.logistic(x) > 0.5:
            return_value = 1
        else:
            return_value = 0

        return return_value

    # TODO: Implement a single gradient ascent step, and return the objective (*not* the log-likelihood!!)
    # Question: What is the objective?
    def step(self, dat, lbl, lr):

        curr_W1, curr_W2, curr_b = self.W1.copy(), self.W2.copy(), self.b

        d = dat.shape[-1]
        gradient_W1 = nr.randn(d, d)
        gradient_W2 = nr.randn(d)
        gradient_b = nr.randn()

        for i in range(len(dat)):
            gradient_W1 += self.dLLdW1(dat[i]) * (lbl[i] - self.logistic(dat[i]))
            gradient_W2 += self.dLLdW2(dat[i]) * (lbl[i] - self.logistic(dat[i]))
            gradient_b += self.dLLdb(dat[i]) * (lbl[i] - self.logistic(dat[i]))

        next_W1 = curr_W1 + lr * gradient_W1
        next_W2 = curr_W2 + lr * gradient_W2
        next_b = curr_b + lr * gradient_b

        self.W1, self.W2, self.b = next_W1, next_W2, next_b

        loss = self.l2_reg() - self.log_likelihood(dat, lbl)
        return loss

    # TODO: Implement the F1 measure computation. 
    def f1(step, test_data, test_gt):
        predictions = [step.predict(x) for x in test_data]

        true_positive = 0
        false_positive = 0
        false_negative = 0
        positive = 0

        for prediction, test in zip(predictions, test_gt):
            if prediction == 1 and test == 1:
                true_positive += 1
            elif prediction == 1 and test == 0:
                false_positive += 1
            elif prediction == 0 and test == 1:
                false_negative += 1

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (false_negative + true_positive)

        f1_score = 2 * (precision * recall) / (precision + recall)

        if f1_score < 0:
            f1_score = 0

        return f1_score


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

    # print("training data shape [-1]: ", tr_data.shape[-1]) # get column number for training data set = 10
    # So training set has 1000 data with 10-dimension vector

    # W1: 10x10, W2: 10x1, b: scalar, lambda: 1e-2 (default)
    # print(tr_data.shape[-1]) # 10-demension
    model = LogReg(tr_data.shape[-1], lamb)
    # x^2+x+c , with lamb

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

    loss_list = []
    f1_list = []
    val_loss_list = []
    val_f1_list = []

    for ep in range(num_epoch):

        val_loss = model.l2_reg() - model.log_likelihood(va_data, va_gt)
        val_loss_list.append(val_loss)

        val_f1 = model.f1(va_data, va_gt)
        val_f1_list.append(val_f1)

        loss = model.step(tr_data, tr_gt, learning_rate)
        loss_list.append(loss)

        f1 = model.f1(te_data, te_gt)
        f1_list.append(f1)

        # Maybe add your own learning rate scheduler here?
        print('[Epoch {}] Regularized loss = {}'.format(ep, loss))

    print('F1 score on test data = {}'.format(model.f1(te_data, te_gt)))

    # PLOT Loss and F1
    plt.plot(range(num_epoch), loss_list, color='green', label='Training Loss')
    plt.plot(range(num_epoch), val_loss_list, color='orange', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epoch')
    plt.show()

    # PLOT F1 Score
    plt.plot(range(num_epoch), f1_list, color='skyblue', label='Training F1')
    plt.plot(range(num_epoch), val_f1_list, color='black', label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.title('Training and Validation F1 vs Epoch')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambdaValue", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    main(args)
