from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
#########################################################################################################################
######################################## input data set model 222222222 #################################################
# train
input_file = "p3.csv"
dataset = pd.read_csv(input_file, header = 0, delimiter = ",")
dataset = dataset.as_matrix()
dataset = dataset[:,0:5]
print(np.shape(dataset))
nrow = len(dataset) -1
nrow1 = nrow +1
nrow_tr = nrow -500
nrow_tr1 = nrow1 -500
#reshape the data
train_X = dataset[0:nrow_tr,:]
train_Y = dataset[1:nrow_tr1,:]
X_vali = dataset[nrow_tr:len(dataset)-1,:]
y_vali = dataset[nrow_tr1:,:]
print(np.shape(X_vali))
print(np.shape(y_vali))
#X_train, X_vali, y_train, y_vali = train_test_split(train_X,train_Y,test_size=0.20, shuffle=False, stratify = None)

X = np.reshape(train_X,[-1,1,5]) # for lstm: sample, length, features
X_vali = np.reshape(X_vali,[-1,1,5])
Y = np.reshape(train_Y,[-1,5])
y_vali = np.reshape(y_vali,[-1,5])


# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
# """""""""""""""stacked lstm"""""""""""""""

model.add(LSTM(64, return_sequences=True,
               input_shape=(X.shape[1],X.shape[2])))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(64))  # return a single vector of dimension 32

# """""""""""""""normal lstm""""""""""""""""
#model.add(LSTM(64, input_shape=(X.shape[1],X.shape[2])))
# fully connected layers
model.add(Dense(128, activation='linear'))
model.add(Dense(256, activation='linear'))
#model.add(Dense(256, activation='linear'))
model.add(Dense(5, activation='linear'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y,epochs=25)

predictY = model.predict(X_vali)
x_axis = range(0,len(y_vali),1)
plt.plot(x_axis, predictY[:,0],'r-',x_axis, y_vali[:,0], 'b')
plt.show()
plt.plot(x_axis, predictY[:,1],'r-',x_axis, y_vali[:,1], 'b')
plt.show()
plt.plot(x_axis, predictY[:,2],'r-',x_axis, y_vali[:,2], 'b')
plt.show()
plt.plot(x_axis, predictY[:,3],'r-',x_axis, y_vali[:,3], 'b')
plt.show()
plt.plot(x_axis, predictY[:,4],'r-',x_axis, y_vali[:,4], 'b')
plt.show()
