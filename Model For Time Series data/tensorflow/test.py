# Simple example using recurrent neural network to predict time series values

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.normalization import batch_normalization
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

step_radians = 0.01
steps_of_history = 100
steps_in_future = 2
index = 0

x = np.arange(0, 4*math.pi, step_radians)
y = np.sin(x)

# Put the data into the right shape
while (index+steps_of_history+steps_in_future < len(y)):
    window = y[index:index+steps_of_history]
    target = y[index+steps_of_history+steps_in_future]
    if index == 0:
        trainX = window
        trainY = target
    else:
        trainX = np.vstack([trainX, window])
        trainY = np.append(trainY, target)
    index = index+1
trainX.shape = (index, steps_of_history, 1)
trainY.shape = (index, 1)

# Network building
net = tflearn.input_data(shape=[None, steps_of_history, 1])
net = tflearn.simple_rnn(net, n_units=512, return_seq=False)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 1, activation='linear')
net = tflearn.regression(net, optimizer='sgd', loss='mean_square', learning_rate=0.001)

# Training
model = tflearn.DNN(net, clip_gradients=0.0, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=15, validation_set=0.1, show_metric=True, batch_size=128)

# Prepare the testing data set
# testX = window to use for prediction
# testY = actual value
# predictY = predicted value
index = 0
while (index+steps_of_history+steps_in_future < len(y)):
    window = y[index:index+steps_of_history]
    target = y[index+steps_of_history+steps_in_future]
    if index == 0:
        testX = window
        testY = target
    else:
        testX = np.vstack([testX, window])
        testY = np.append(testY, target)
    index = index+1
testX.shape = (index, steps_of_history, 1)
testY.shape = (index, 1)

# Predict the future values
predictY = model.predict(testX)

# Plot the results
plt.figure(figsize=(20,4))
plt.suptitle('Prediction')
plt.title('History='+str(steps_of_history)+', Future='+str(steps_in_future))
plt.plot(y, 'r-', label='Actual')
plt.plot(predictY, 'gx', label='Predicted')
plt.legend()
plt.savefig('sine.png')
