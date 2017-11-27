import tflearn
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing

# """"""""""""""""define new function""""""""""""""""
# RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# """"""""""""""""""""""input data set""""""""""""""""""""""
# train
input_file = "dataset_angle.csv"
dataset = pd.read_csv(input_file, header = 0, delimiter = ";")
dataset = dataset.as_matrix()
dataset = dataset[:,5]/0.04
# 1:3.0 ,2: 0.75 ,3:0.07 (better *5) ,4:0.12 (better *10), 5:0.04
# """"""""""""""""" preprocessing data """"""""""""""""""""
#min_max_scaler = preprocessing.MinMaxScaler()
#dataset = min_max_scaler.fit_transform(dataset)

# test
input_test = "testset.csv"
testset = pd.read_csv(input_test, header = 0, delimiter = ",")
testset = testset.as_matrix()
testset = testset[:,4]*5
# add noise to testdata
#test_noise = np.random.normal(0,0.1,99)

# """""""""""""""""""""""""define training"""""""""""""""""""""""

train_X = np.delete(dataset, 1435)
#train_X = train_X + train_noise
train_X = np.reshape(train_X,[-1, 1])
X = np.split(train_X,1435, 0)
train_Y = np.delete(dataset, 0)
#train_Y = train_Y + train_noise
Y = np.reshape(train_Y,[-1,1])
"""
train_X = dataset[0:1435,:]
#train_X = train_X + train_noise
train_X = np.reshape(train_X,[-1, 1])
X = np.split(train_X,1435, 0)
train_Y = dataset[1:1436,:]
#train_Y = train_Y + train_noise
Y = np.reshape(train_Y,[-1,1])
"""
# """"""""""""""""""""""""Network building""""""""""""""""""""""""
net = tflearn.input_data(shape=[None,1,1])
#net = tflearn.embedding(net, input_dim=2, output_dim=3)  for text prediction
net = tflearn.lstm(net, 256, activation='relu')
net = tflearn.fully_connected(net, 1, activation='linear')
#net = rflearn.dropout(net, 0.8)
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='mean_square')

# """""""""""""""""""""""Training"""""""""""""""""""""""
model = tflearn.DNN(net, clip_gradients=0.0, tensorboard_verbose=0)
model.fit(X, Y, show_metric=True, n_epoch=15, validation_set=0.2)
# show the predicted and actual value of training
predictY = model.predict(X)
x_axis = range(0,1435,1)
plt.plot(x_axis, predictY,'r-',x_axis, train_Y, 'b')
plt.show()

# """"""""""""""""""""""""testing"""""""""""""""""""""""
"""
# define testset
test_X = np.delete(testset, 99)
test_X = test_X + test_noise
test_X = np.reshape(test_X,[-1, 1])
test_X = np.split(test_X,99, 0)
test_Y = np.delete(testset, 0)
test_Y = test_Y + test_noise
test_Y = np.reshape(test_Y,[-1,1])
# predict the unseen value with the actual value
predictY = model.predict(test_X)
x_axis = range(0,99,1)
plt.plot(x_axis, predictY,'r-',x_axis, test_Y, 'b')
plt.show()
# """"""""""""""""""evaluate the Prediction"""""""""""""""""""
print(rmse(predictY,test_Y))
"""
