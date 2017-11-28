import tensorflow as tf
import numpy as np
import pandas as pd

##### """""""""""""""input the dataset into model """"""""""""""""
input_file = "dataset.csv"
input_file_test = "testdata.csv"

# for space delimited use:
#df = pd.read_csv(input_file, header = 0, delimiter = ";")
df_test = pd.read_csv(input_file_test, header = 0, delimiter = ",")

# create a numpy array with the numeric values for input into scikit-learn
# process training data

# process testdata
numpy_array_test = df_test.as_matrix()


# get pair
def get_pair():
    """
    Returns an (current, later) pair, where 'later' is 'lag'
    steps ahead of the 'current' on the wave(s) as defined by the
    frequency.
    """

    global sliding_window
    sliding_window.append(get_sample())
    input_value = sliding_window[0]
    output_value = sliding_window[-1]
    sliding_window = sliding_window[1:]
    return input_value, output_value


#Imports
import tensorflow as tf

#Input Params
input_dim = 1

##The Input Layer as a Placeholder
#Since we will provide data sequentially, the 'batch size'
#is 1.
input_layer = tf.placeholder(tf.float32, [1, input_dim])

##The LSTM Layer-1
#The LSTM Cell initialization
lstm_layer1 = tf.nn.rnn_cell.BasicLSTMCell(input_dim)
#The LSTM state as a Variable initialized to zeroes
lstm_state1 = tf.Variable(tf.zeros([1, lstm_layer1.state_size]))
#Connect the input layer and initial LSTM state to the LSTM cell
lstm_output1, lstm_state_output1 = lstm_layer1(input_layer, lstm_state1)
#The LSTM state will get updated
lstm_update_op1 = lstm_state1.assign(lstm_state_output1)

##The Regression-Output Layer1
#The Weights and Biases matrices first
output_W1 = tf.Variable(tf.truncated_normal([input_dim, input_dim]))
output_b1 = tf.Variable(tf.zeros([input_dim]))
#Compute the output
final_output = tf.matmul(lstm_output1, output_W1) + output_b1

##Input for correct output (for training)
correct_output = tf.placeholder(tf.float32, [1, input_dim])

##Calculate the Sum-of-Squares Error
error = tf.pow(tf.sub(final_output, correct_output), 2)

##The Optimizer
#Adam works best
train_step = tf.train.AdamOptimizer(0.0006).minimize(error)

##Session
sess = tf.Session()
#Initialize all Variables
sess.run(tf.initialize_all_variables())

##Training

actual_output1 = []
network_output1 = []
x_axis = []


for i in range(80000):
    input_v, output_v = get_total_input_output()
    _, _, network_output = sess.run([lstm_update_op1,
                                     train_step,
                                     final_output],
                                    feed_dict = {
                                        input_layer: input_v,
                                        correct_output: output_v})

    actual_output1.append(output_v[0])
    network_output1.append(network_output[0])
    x_axis.append(i)

import matplotlib.pyplot as plt
plt.plot(x_axis, network_output1, 'r-', x_axis, actual_output1, 'b-')
plt.show()
