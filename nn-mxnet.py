# Tx = Tx-1, Tx-2 ; Window size = 2

# Imports for running this file
import mxnet as mx
import numpy as np
import data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load needed data
rawData = data.loadPickle("Opendata Bahn/stationsListFfmTotal.pkl")
rawData.head(1)

dataset = rawData['total'].values
ticks = range(len(dataset))

# Reshaping  and scaling dataset for fitting into neural network
dataset = np.reshape(dataset, (len(dataset), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scaler.fit_transform(dataset)

# Creating data variables with sliding window shape
seq_len = 2

x = scaled_dataset
y = scaled_dataset[:, [-1]]

dataX = []
dataY = []

for i in range(0, len(y)-seq_len):
    _x = x[i: i+seq_len]
    _y = y[i+seq_len]
    dataX.append(_x)
    dataY.append(_y)


# Split the data in test and train
train_size = int(len(dataY) * 0.6)
test_size = len(dataY) - train_size

batch_size = 1

trainX, testX = np.array(dataX[:train_size]), np.array(dataX[train_size:])
trainY, testY = np.array(dataY[:train_size]), np.array(dataY[train_size:])

train_iter = mx.io.NDArrayIter(data=trainX, label=trainY,
                               batch_size=batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data=testX, label=testY,
                             batch_size=batch_size, shuffle=False)


# Lets build the network
data = mx.sym.var("data")
data = mx.sym.transpose(data, axes=(1, 0, 2))

# T N C -- Time Steps/ Seq len; N - Batch Size, C - dimensions in the hidden state


lstm1 = mx.rnn.LSTMCell(num_hidden=5, prefix='lstm1')
lstm2 = mx.rnn.LSTMCell(num_hidden=10, prefix='lstm2')

L1, states = lstm1.unroll(length=seq_len, inputs=data,
                          merge_outputs=True, layout="TNC")
L2, L2_states = lstm2.unroll(
    length=seq_len, inputs=L1, merge_outputs=True, layout="TNC")

# (T*N, 10 -- num_hidden lstm2)
L2_reshape = mx.sym.reshape(L2_states[0], shape=(-1, 0), reverse=True)
fc = mx.sym.FullyConnected(L2_reshape, num_hidden=1, name='fc')
net = mx.sym.LinearRegressionOutput(data=fc, name="softmax")


num_epochs = 10
model = mx.mod.Module(symbol=net, context=mx.cpu(0))

model.fit(train_data=train_iter, eval_data=val_iter,
          optimizer="adam",
          optimizer_params={'learning_rate': 1E-3},
          eval_metric="mse",
          num_epoch=num_epochs
          )


def myr2(T, Y):
    Ym = T.mean()
    sse = (T - Y).dot(T - Y)
    sst = (T - Ym).dot(T - Ym)
    return 1 - sse / sst


test_pred = model.predict(val_iter).asnumpy()
print(np.mean((test_pred - testY)**2), myr2(testY[:, 0], test_pred[:, 0]))


test_plot = scaler.inverse_transform(test_pred)

t_plot = np.empty_like(dataset)
t_plot[:] = np.nan
t_plot[len(trainY): -seq_len] = test_plot
t_plot


len(test_plot), len(testY), len(trainY), len(rawData), ticks[0:10]
idx = []
for i in ticks:
    idx.append(i)

idx2 = idx[:(len(idx)-2)]

train_plot = scaler.inverse_transform(trainY)
test_foo = scaler.inverse_transform(testY)


dataIndex = pd.date_range('1/1/2014', periods=1232, freq='D')
dataFrame = pd.DataFrame(idx, index=dataIndex)

# Plot with mathplotlib. Compare with other results
plt.figure(figsize=(15, 5))
plt.plot(idx, dataset, label="True value")
plt.plot(idx2[:len(trainY)], train_plot[:, 0], label="Training set prediction")
plt.plot(idx2[len(trainY):], test_plot[:, 0], label="Test set prediction")
plt.xlabel("Days")
plt.ylabel("Number of rented bikes a day")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()

plt.close()
