from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# """"""""""""""""""""""input data set""""""""""""""""""""""
# train
input_file = "pattern7.csv"
dataset = pd.read_csv(input_file, header = 0, delimiter = ",")
dataset = dataset.as_matrix()
dataset = dataset[:,0:4]
print np.shape(dataset)
nrow = len(dataset) -1
nrow1 = nrow +1
nrow_tr = nrow -100
nrow_tr1 = nrow1 -100
#reshape the data
train_X = dataset[0:nrow_tr,:]
train_Y = dataset[1:nrow_tr1,:]
X_vali = dataset[nrow_tr:len(dataset)-1,:]
y_vali = dataset[nrow_tr1:,:]
print(np.shape(X_vali))
print(np.shape(y_vali))
X = np.reshape(train_X,[-1,1,4]) # for lstm: sample, length, features
X_vali = np.reshape(X_vali,[-1,1,4])
Y = np.reshape(train_Y,[-1,4])
y_vali = np.reshape(y_vali,[-1,4])
########### print data prediction structure ###############
"""
predictY = model.predict(X_vali)
x_axis = range(0,len(y_vali)-1,1)
plt.plot(x_axis, predictY[:-1,0],'r-',x_axis, y_vali[1:,0], 'b')
plt.show()
plt.plot(x_axis, predictY[:,1],'r-',x_axis, y_vali[:,1], 'b')
plt.show()
plt.plot(x_axis, predictY[:,2],'r-',x_axis, y_vali[:,2], 'b')
plt.show()
plt.plot(x_axis, predictY[:,3],'r-',x_axis, y_vali[:,3], 'b')
plt.show()
plt.plot(x_axis, predictY[:,4],'r-',x_axis, y_vali[:,4], 'b')
plt.show()
"""
################### loading model 1 #######################
model = load_model('model_1.h5')
predictY = model.predict(X_vali)
p_err1 = predictY[:-1,] - y_vali[1:,]
p_err1 = np.sum(p_err1**2)
print("pattern 1 error:",p_err1)
################### loading model 2 #######################
model = load_model('model_2.h5')
predictY = model.predict(X_vali)
p_err2 = predictY[:-1,] - y_vali[1:,]
p_err2 = np.sum(p_err2**2)
print("pattern 2 error:",p_err2)
################### loading model 3 #######################
model = load_model('model_3.h5')
predictY = model.predict(X_vali)
p_err3 = predictY[:-1,] - y_vali[1:,]
p_err3 = np.sum(p_err3**2)
print("pattern 3 error:",p_err3)
################### loading model 4 #######################
model = load_model('model_4.h5')
predictY = model.predict(X_vali)
p_err4 = predictY[:-1,] - y_vali[1:,]
p_err4 = np.sum(p_err4**2)
print("pattern 4 error:",p_err4)
################### loading model 4 #######################
model = load_model('model_5.h5')
predictY = model.predict(X_vali)
p_err5 = predictY[:-1,] - y_vali[1:,]
p_err5 = np.sum(p_err5**2)
print("pattern 5 error:",p_err5)
################### loading model 4 #######################
model = load_model('model_6.h5')
predictY = model.predict(X_vali)
p_err6 = predictY[:-1,] - y_vali[1:,]
p_err6 = np.sum(p_err6**2)
print("pattern 6 error:",p_err6)
################### loading model 4 #######################
model = load_model('model_7.h5')
predictY = model.predict(X_vali)
p_err7 = predictY[:-1,] - y_vali[1:,]
p_err7 = np.sum(p_err7**2)
print("pattern 7 error:",p_err7)
#**********************************************************#
#********************* RECOGNITION RESULTS*****************#

d = {"pattern 1":p_err1, "pattern 2":p_err2, "pattern 3":p_err3,
     "pattern 4":p_err4, "pattern 5":p_err5, "pattern 6":p_err6,
     "pattern 7":p_err7}
print(min(d, key=lambda k: d[k]))
