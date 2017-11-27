from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
#########################################################################################################################
######################################## input data set model 111111111 #################################################
# train
input_file = "pattern6.csv"
dataset = pd.read_csv(input_file, header = 0, delimiter = ";")
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
#X_train, X_vali, y_train, y_vali = train_test_split(train_X,train_Y,test_size=0.20, shuffle=False, stratify = None)

X = np.reshape(train_X,[-1,1,4]) # for lstm: sample, length, features
X_vali = np.reshape(X_vali,[-1,1,4])
Y = np.reshape(train_Y,[-1,4])
y_vali = np.reshape(y_vali,[-1,4])


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
model.add(Dense(4, activation='linear'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y,epochs=50)
from keras.models import load_model

# """""""""""""""" save the structure of the model in .json file """"""""""""""
model_structure = model.to_json()

# """"""""""""""" save model 111111111 """"""""""""""""
model.save('model_6.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

############################################### END MODEL 1 ###################################################
###############################################################################################################

#
