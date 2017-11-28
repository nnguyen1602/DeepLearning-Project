import tensorflow as tf
import numpy as np
import pandas as pd

##### """""""""""""""input the dataset into model """"""""""""""""
input_file = "dataset.csv"
input_file_test = "testdata.csv"

# for space delimited use:
df = pd.read_csv(input_file, header = 0, delimiter = ";")
df_test = pd.read_csv(input_file_test, header = 0, delimiter = ",")

# create a numpy array with the numeric values for input into scikit-learn
numpy_array = df.as_matrix()
numpy_array_test = df_test.as_matrix()

input_data = numpy_array[:,1]
input_data_test = numpy_array[:,1]
#print(np.shape(input_data))

##### """"""""""""""" setup parameteres """""""""""""""
num_inputs = 1  # input dimension
num_outputs = 1 # output dimension

num_steps = 1396
num_steps_test = 99
num_hidden = 5    # use 5 cells in hidden layer
num_epochs = 10  # 100 iterations
batch_size = 1     # only one sine wave per batch


def gen_data(distort=True):
    input_data_ = np.delete(input_data, 1396)
    X = input_data_.reshape(num_steps,1)
    if distort:
        X = np.add(X, np.random.uniform(-0.1,0.1,size=[num_steps,1]))

    y = np.delete(input_data, 0)

    X = X.reshape(batch_size,num_steps,1)
    y = y.reshape(batch_size,num_steps,1)

    return (X,y)


def gen_data_test(distort=True):
    input_data_test_ = np.delete(input_data_test, 100)
    X_test = input_data_test_.reshape(num_steps_test,1)
    if distort:
        X_test = np.add(X_test, np.random.uniform(-0.1,0.1,size=[num_steps_test,1]))

    y_test = np.delete(input_data_test, 0)

    X_test = X_test.reshape(batch_size,num_steps_test,1)
    y_test = y_test.reshape(batch_size,num_steps_test,1)

    return (X_test,y_test)


def create_model():
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_hidden)
    # The LSTM state as a Variable initialized to zeroes
    #lstm_state = tf.Variable(tf.zeros([1, cell.state_size]))

    inputs = tf.placeholder(shape=[batch_size, num_steps,num_inputs],dtype=tf.float32)
    result = tf.placeholder(shape=[batch_size, num_steps, 1],dtype=tf.float32)

    X = tf.transpose(inputs,[1,0,2]) # num_steps (T) elements of shape (batch_size x num_inputs)
    X = tf.reshape(X,[-1, num_inputs]) # flatten the data with num_inputs values on each row

    X = tf.split(X,num_steps, 0) # create a list with an element per timestep, cause that is what the rnn needs as input
    resultY = tf.transpose(result,[1,0,2]) # swap the first two dimensions, in order to be compatible with the input args

        #print(X)

    outputs,states = tf.nn.static_rnn(cell, inputs=X, dtype=tf.float32) # initial_state=init_state) # outputs & states for each time step
        #The LSTM state will get updated
    lstm_state_update = states

    w_output = tf.Variable(tf.random_normal([num_steps, num_hidden], stddev=0.01, dtype=tf.float32))
    b_output = tf.Variable(tf.random_normal([num_steps, 1], stddev=0.01, dtype=tf.float32))

    the_output = []



    for i in range(num_steps):

        #print(outputs[i])
        #print(w_output[i:i+1,:])
        #print(tf.matmul(outputs[i],w_output[i:i+1,:],transpose_b=True))

        # print (the_output[i])

        #print ( tf.nn.sigmoid(tf.matmul(outputs[i], w_output[i:i+1,:], transpose_b=True)) )
        the_output.append(tf.matmul(outputs[i], w_output[i:i+1,:], transpose_b=True)+ b_output[i])



    outputY = tf.stack(the_output)

    cost = tf.reduce_mean(tf.pow(outputY - resultY,2))
        #cross_entropy = -tf.reduce_sum(resultY * tf.log(tf.clip_by_value(outputY,1e-10,1.0)))

        #train_op = tf.train.RMSPropOptimizer(0.005,0.2).minimize(cost)
        #train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    train_op = tf.train.AdamOptimizer(0.01).minimize(cost)

    valX,valy = gen_data(False)  # validate with clean sine wave
     # validate with clean sine wave

    with tf.Session() as sess:

        #print("gen data")

        #tf.merge_all_summaries()
        #writer = tf.train.SummaryWriter("/Users/joost/tensorflow/log",flush_secs=10,graph_def=sess.graph_def)
        tf.initialize_all_variables().run()

        ##### create Drawing parameters

        x_axis = range(0,num_steps,1)
        #costval_container = []
        predict_container = []
        real_container = []


        for k in range(num_epochs):

            # print("start k={}".format(k))
            tempX,y = gen_data()   # tempX has batch_size elements of shape ( num_steps x num_inputs)
            dict = {inputs: tempX, result: y}
            sess.run(train_op, feed_dict=dict)
            valdict = {inputs: valX, result: valy}
            updated_state,costval,outputval = sess.run((lstm_state_update,cost,outputY), feed_dict=dict)
            #print(updated_state)
            print(outputval[:,0,0] - y[0,:,0])
            #costval_container.append(costval)
            #real_container
            #print(costval)
        #print(outputval)
        #print(costval_container)
        #print(updated_state)
        #print(np.shape(outputval)) # shape: (1396, 1, 1)
        #print(np.shape(y))         # shape: (1, 1396, 1)
        #print(np.shape(costval))   #

        import matplotlib.pyplot as plt
        plt.plot(x_axis, outputval[:,0,0], 'r-', x_axis, y[0,:,0], 'b-')
        plt.show()
    return updated_state

create_model()
        ################# Testing process ####################
"""
def test():
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_hidden)

    # The LSTM state as a Variable initialized to zeroes
    #lstm_state = tf.Variable(tf.zeros([1, cell.state_size]))

    test_inputs = tf.placeholder(shape=[batch_size, num_steps_test,num_inputs],dtype=tf.float32)
    test_result = tf.placeholder(shape=[batch_size, num_steps_test, 1],dtype=tf.float32)

    test_X = tf.transpose(test_inputs,[1,0,2]) # num_steps (T) elements of shape (batch_size x num_inputs)
    test_X = tf.reshape(test_X,[-1, num_inputs]) # flatten the data with num_inputs values on each row

    test_X = tf.split(test_X,num_steps_test, 0) # create a list with an element per timestep, cause that is what the rnn needs as input
    test_resultY = tf.transpose(test_result,[1,0,2]) # swap the first two dimensions, in order to be compatible with the input args

        #print(X)

    test_outputs,test_states = tf.nn.static_rnn(cell, inputs=test_X, dtype=tf.float32) # initial_state=init_state) # outputs & states for each time step
        #The LSTM state will get updated
    test_lstm_state_update = test_states

    w_output_test = tf.Variable(tf.random_normal([num_steps_test, num_hidden], stddev=0.01, dtype=tf.float32))
    b_output_test = tf.Variable(tf.random_normal([num_steps_test, 1], stddev=0.01, dtype=tf.float32))

    the_output_test = []


    for i in range(num_steps):
        the_output_test.append(tf.matmul(test_outputs[i], w_output_test[i:i+1,:], transpose_b=True)+ b_output_test[i])

    test_outputY = tf.stack(the_output_test)
    test_valX,test_valy = gen_data_test(False)

    with tf.Session() as sess:

        tf.initialize_all_variables().run()

        x_test, y_test = gen_data_test()   # tempX has batch_size elements of shape ( num_steps x num_inputs)
        dict_test = {inputs: x_test, result: y_test}
        sess.run(train_op, feed_dict=dict_test)
        valdict = {inputs: test_valX, result: test_valy}
        test_predict = sess.run(test_outputY, feed_dict=dict_test)
        #print(updated_state)
        print(test_predict)

test()
"""
