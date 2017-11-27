import tflearn
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tflearn.data_preprocessing import DataPreprocessing

# """"""""""""""""define new function""""""""""""""""
# RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# """"""""""""""""data preprocessing"""""""""""""""""
Data_prep = DataPreprocessing()
#Data_prep.add_featurewise_zero_center(mean=0.5)
#Data_prep.add_featurewise_stdnorm()
# """"""""""""""""""""""input data set""""""""""""""""""""""
# train
input_file = "hieu.csv"
dataset = pd.read_csv(input_file, header = 0, delimiter = ";")
dataset = dataset.as_matrix()
dataset = dataset[:,1:6]
min_max_scaler = MinMaxScaler()
#""""""""""""""""""""""""""""""""""""""""""""
data_01 = np.full(2354, np.amin(dataset[:,0]))
data_02 = np.amax(dataset[:,0])-np.amin(dataset[:,0])
dataset[:,0] = (dataset[:,0] - np.full(2354, np.amin(dataset[:,0])))/(np.amax(dataset[:,0])-np.amin(dataset[:,0]))
#""""""""""""""""""""""""""""""""""""""""""""
data_11 = np.full(2354, np.amin(dataset[:,1]))
data_12 = np.amax(dataset[:,1])-np.amin(dataset[:,1])
dataset[:,1] = (dataset[:,1] - np.full(2354, np.amin(dataset[:,1])))/(np.amax(dataset[:,1])-np.amin(dataset[:,1]))
#""""""""""""""""""""""""""""""""""""""""""""
data_21 = np.full(2354, np.amin(dataset[:,2]))
data_22 = np.amax(dataset[:,2])-np.amin(dataset[:,2])
dataset[:,2] = (dataset[:,2] - np.full(2354, np.amin(dataset[:,2])))/(np.amax(dataset[:,2])-np.amin(dataset[:,2]))
#""""""""""""""""""""""""""""""""""""""""""""
data_31 = np.full(2354, np.amin(dataset[:,3]))
data_32 = np.amax(dataset[:,3])-np.amin(dataset[:,3])
dataset[:,3] = (dataset[:,3] - np.full(2354, np.amin(dataset[:,3])))/(np.amax(dataset[:,3])-np.amin(dataset[:,3]))
#""""""""""""""""""""""""""""""""""""""""""""
data_41 = np.full(2354, np.amin(dataset[:,4]))
data_42 = np.amax(dataset[:,4])-np.amin(dataset[:,4])
dataset[:,4] = (dataset[:,4] - np.full(2354, np.amin(dataset[:,4])))/(np.amax(dataset[:,4])-np.amin(dataset[:,4]))

# 1:3.0 ,2: 0.75 ,3:0.07 (better *5) ,4:0.12 (better *10), 5:0.04
# add noise to traindata
#train_noise = np.random.normal(0,0.3,1396)

"""
# test
input_test = "testset.csv"
testset = pd.read_csv(input_test, header = 0, delimiter = ",")
testset = testset.as_matrix()
testset = testset[:,1]
# add noise to testdata
test_noise = np.random.normal(0,0.1,99)
"""
# """""""""""""""""""""""""define training"""""""""""""""""""""""
train_X = dataset[0:2353,:]
#train_X = train_X + train_noise
train_X = np.reshape(train_X,[-1, 5])
#train_X = min_max_scaler.fit_transform(train_X)
X = np.split(train_X,2353, 0)
train_Y = dataset[1:2354,:]
#train_Y = train_Y + train_noise
Y = np.reshape(train_Y,[-1,5])


# """"""""""""""""""""""""Network building""""""""""""""""""""""""
net = tflearn.input_data(shape=[None,1,5])
#net = tflearn.embedding(net, input_dim=2, output_dim=3)  for text prediction
net = tflearn.lstm(net, 256, activation='linear')
#net = tflearn.fully_connected(net, 128, activation='linear')
#net = tflearn.fully_connected(net, 256, activation='linear')
net = tflearn.fully_connected(net, 5, activation='linear') # correct f(x)=x
#net = tflearn.dropout(net, 0.8)
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='mean_square')

# """""""""""""""""""""""Training"""""""""""""""""""""""
model = tflearn.DNN(net, clip_gradients=0.0, tensorboard_verbose=0)
model.fit(X, Y, show_metric=True, n_epoch=30, validation_set=0.2)
# show the predicted and actual value of training
predictY = model.predict(X)
x_axis = range(0,2353,1)
plt.plot(x_axis, predictY[:,0],'r-',x_axis, train_Y[:,0], 'b')
plt.show()
plt.plot(x_axis, predictY[:,1],'r-',x_axis, train_Y[:,1], 'b')
plt.show()
plt.plot(x_axis, predictY[:,2],'r-',x_axis, train_Y[:,2], 'b')
plt.show()
plt.plot(x_axis, predictY[:,3],'r-',x_axis, train_Y[:,3], 'b')
plt.show()
plt.plot(x_axis, predictY[:,4],'r-',x_axis, train_Y[:,4], 'b')
plt.show()

# """"""""""""""""""""""""testing"""""""""""""""""""""""
# define testset
"""
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
