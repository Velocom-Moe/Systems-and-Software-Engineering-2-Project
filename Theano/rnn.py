# The corresponding tutorial for this code was released EXCLUSIVELY as a bonus
# If you want to learn about future bonuses, please sign up for my newsletter at:
# https://lazyprogrammer.me

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


import os
import sys

from lstm import LSTM
from gru import GRU


def init_weight(M1, M2):
    return np.random.randn(M1, M2) / np.sqrt(M1 + M2)


def myr2(T, Y):
    Ym = T.mean()
    sse = (T - Y).dot(T - Y)
    sst = (T - Ym).dot(T - Ym)
    return 1 - sse / sst


class RNN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, activation=T.tanh, learning_rate=1e-1, mu=0.5, reg=0, epochs=120, show_fig=False):
        N, t, D = X.shape

        self.hidden_layers = []
        Mi = D
        for Mo in self.hidden_layer_sizes:
            ru = GRU(Mi, Mo, activation)
            self.hidden_layers.append(ru)
            Mi = Mo

        Wo = np.random.randn(Mi) / np.sqrt(Mi)
        bo = 0.0
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wo, self.bo]
        for ru in self.hidden_layers:
            self.params += ru.params

        lr = T.scalar('lr')
        thX = T.matrix('X')
        thY = T.scalar('Y')
        Yhat = self.forward(thX)[-1]

        # let's return py_x too so we can draw a sample instead
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=Yhat,
            allow_input_downcast=True,
        )

        cost = T.mean((thY - Yhat) * (thY - Yhat))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value() * 0) for p in self.params]

        updates = [
                      (p, p + mu * dp - lr * g) for p, dp, g in zip(self.params, dparams, grads)
                  ] + [
                      (dp, mu * dp - lr * g) for dp, g in zip(dparams, grads)
                  ]

        self.train_op = theano.function(
            inputs=[lr, thX, thY],
            outputs=cost,
            updates=updates
        )

        costs = []
        for i in range(epochs):
            t0 = datetime.now()
            X, Y = shuffle(X, Y)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(N):
                c = self.train_op(learning_rate, X[j], Y[j])
                cost += c
            if i % 10 == 0:
                print(                "i:", i, "cost:", cost, "time for epoch:", (datetime.now() - t0))
            if (i + 1) % 500 == 0:
                learning_rate /= 10
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.output(Z)
        return Z.dot(self.Wo) + self.bo

    def score(self, X, Y):
        Yhat = self.predict(X)
        return myr2(Y, Yhat)

    def predict(self, X):
        N = len(X)
        Yhat = np.empty(N)
        for i in range(N):
            Yhat[i] = self.predict_op(X[i])
        return Yhat


# we need to skip the 3 footer rows
# skipfooter does not work with the default engine, 'c'
# so we need to explicitly set it to 'python'
#df = pd.read_csv('international-airline-passengers.csv', engine='python', skipfooter=3)
df = pd.read_csv("../Opendata Bahn/stationsListFfmTotal.csv",
                      #usecols = [1],
                      engine = "python",
                      skipfooter = 3)

# rename the columns because they are ridiculous
df.columns = ['month', 'num_passengers']

# plot the data so we know what it looks like
# plt.plot(df.num_passengers)
# plt.show()

# let's try with only the time series itself
series = df.num_passengers.as_matrix()
# series = (series - series.mean()) / series.std() # normalize the values so they have mean 0 and variance 1
series = series.astype(np.float32)
series = series - series.min()
series = series / series.max()

# let's see if we can use D past values to predict the next value
N = len(series)
D = 4
n = N - D
X = np.zeros((n, D))
for d in range(D):
    X[:, d] = series[d:d + n]
Y = series[D:D + n]

print(    "series length:", n)
split_index = int(round((n/10)*6))
Xtrain = X[:split_index]
Ytrain = Y[:split_index]
Xtest = X[split_index:]
Ytest = Y[split_index:]

Ntrain = len(Xtrain)
Xtrain = Xtrain.reshape(Ntrain, D, 1)
Ntest = len(Xtest)
Xtest = Xtest.reshape(Ntest, D, 1)

model = RNN([50])
model.fit(Xtrain, Ytrain, activation=T.tanh)
print(    "train score:", model.score(Xtrain, Ytrain))
print(    "test score:", model.score(Xtest, Ytest))
YHat = model.predict(Xtrain)
print(np.mean((YHat - Ytrain)**2))
YHat = model.predict(Xtest)
print(np.mean((YHat - Ytest)**2))


plt.figure(figsize=(15, 5))
# plot the prediction with true values
plt.plot(series, label="True value")

train_series = np.empty(n)
train_series[:split_index] = model.predict(Xtrain)
train_series[split_index:] = np.nan
# prepend d nan's since the train series is only of size N - D
plt.plot(np.concatenate([np.full(d, np.nan), train_series]), label="Training set prediction")

test_series = np.empty(n)
test_series[:split_index] = np.nan
test_series[split_index:] = model.predict(Xtest)
plt.plot(np.concatenate([np.full(d, np.nan), test_series]), label="Test set prediction")

plt.xlabel("Days")
plt.ylabel("Number of rented bikes a day")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()

#test_foo = scaler.inverse_transform(test_series)