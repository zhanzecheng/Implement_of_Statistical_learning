import pandas as pd
import numpy as np
from logicalRegression import LogisticRegression
import matplotlib.pyplot as plt

""" HELPER FUNCTION: GET ERROR RATE ========================================="""


def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))


""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""


def print_error_rate(err):
    print('Error rate: Training: %.4f - Test: %.4f' % err)


""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""


def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


""" ADABOOST IMPLEMENTATION ================================================="""

def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights
        weight = np.expand_dims(w, axis=1)

        clf.fit(X_train, Y_train, weight)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Error
        err_m = np.dot(w, miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m+1e-15))
        # New weights
        # print(miss[0], Y_train[0])
        # quit()
        w = np.multiply(w, np.exp([float(pred_train_i[x]) * (-1) * Y_train[x] * alpha_m for x in range(len(miss))]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


""" PLOT FUNCTION ==========================================================="""


def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue'], grid=True)
    plot1.set_xlabel('Number of iterations', fontsize=12)
    plot1.set_xticklabels(range(0, 50, 5))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of ensemble', fontsize=16)
    plt.axhline(y=er_test[0], linewidth=1, color='red', ls='dashed')
    plt.show()
    plt.savefig('./result.png')

def load_dataset():
    y = []
    X = []
    with open('./train.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            X.append(np.array([eval(x) for x in line[:4]]))
            y.append(eval(line[4]))
    X = np.array(X)
    y = np.array(y)

    X_test = []
    y_test = []
    with open('./test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            X_test.append(np.array([eval(x) for x in line[:4]]))
            y_test.append(eval(line[4]))
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X, y, X_test, y_test


if __name__ == "__main__":

    X, y, X_test, y_test = load_dataset()

    clf = LogisticRegression()

    # Test with different number of iterations
    er_train, er_test = [], []
    x_range = [1, 10, 25, 50]
    for i in x_range:
        print('The ensemble size is %s' % (i))
        er_i = adaboost_clf(y, X, y_test, X_test, i, clf)
        # quit()
        er_train.append(er_i[0])
        er_test.append(er_i[1])

    # plot the error rate of trainset and testset
    plot_error_rate(er_train, er_test)