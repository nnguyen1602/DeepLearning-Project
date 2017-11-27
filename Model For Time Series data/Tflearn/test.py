from __future__ import division
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
input_file = "dataset.csv"
dataset = pd.read_csv(input_file, header = 0, delimiter = ";")
dataset = dataset.as_matrix()
dataset = dataset[:,1:6]        # 1:3.0 ,2: 0.75 ,3:0.07 (better *10) ,4:0.12


train_X = dataset[0:2353,:]
#train_X = train_X + train_noise
train_X = np.reshape(train_X,[-1, 5])
#train_X = min_max_scaler.fit_transform(train_X)
X = np.split(train_X,2353, 0)
train_Y = dataset[1:2354,:]
#train_Y = train_Y + train_noise
Y = np.reshape(train_Y,[-1,5])


# """"""""""""""""" preprocessing data """"""""""""""""""""


print(np.full(2353, np.amin(dataset[:,])))
print(np.amax(dataset[:,])-np.amin(dataset[:,]))
print(dataset)
x_axis = range(0,2353,1)
plt.plot(x_axis, Y[:,0],'r-')
plt.show()
plt.plot(x_axis, Y[:,1],'r-')
plt.show()
plt.plot(x_axis, Y[:,2],'r-')
plt.show()
plt.plot(x_axis, Y[:,3],'r-')
plt.show()
plt.plot(x_axis, Y[:,4],'r-')
plt.show()
