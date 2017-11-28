import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import rnn

##### """""""""""""""input the dataset into model """"""""""""""""
input_file = "dataset.csv"
input_file_test = "testdata.csv"

# for space delimited use:
df = pd.read_csv(input_file, header = 0, delimiter = ";")
df_test = pd.read_csv(input_file_test, header = 0, delimiter = ";")

# create a numpy array with the numeric values for input into scikit-learn
numpy_array = df.as_matrix()
numpy_array_test = df_test.as_matrix()

input_data = numpy_array[:,1]
#input_data_test = numpy_array_test[:,1]

num_inputs = 1  # input dimension
num_outputs = 1 # output dimension

num_steps = 1396
#num_steps = 349
num_steps_test = 100
num_hidden = 3    # use 5 cells in hidden layer
num_epochs = 10  # 100 iterations
batch_size = 1     # only one sine wave per batch
#batch_size = 4



def gen_data():
    x = np.delete(input_data, 1396)
    x = x.reshape(num_steps,1)
    #if distort:
     #   X = np.add(X, np.random.uniform(-0.1,0.1,size=[num_steps,1]))

    y = np.delete(input_data, 0)

    x = x.reshape(batch_size,num_steps,1)
    y = y.reshape(batch_size,num_steps,1)

    return (x,y)


def gen_data_test():
    x_test = np.delete(input_data_test, 100)
    x_test = X_test.reshape(num_steps_test,1)
   # if distort:
   #     X_test = np.add(X_test, np.random.uniform(-0.1,0.1,size=[num_steps_test,1]))

    y_test = np.delete(input_data_test, 0)

    x_test = x_test.reshape(batch_size,num_steps_test,1)
    y_test = y_test.reshape(batch_size,num_steps_test,1)

    return (x_test,y_test)



def recurrent_neural_network():
    lstm_cell = rnn.BasicLSTMCell(num_units=num_hidden)

    x_input = tf.placeholder(shape=[batch_size, num_steps,num_inputs],dtype=tf.float32)
    y_result = tf.placeholder(shape=[batch_size, num_steps, num_outputs],dtype=tf.float32)

    x = tf.transpose(x_input, [1,0,2])
    x = tf.reshape(x,[-1, num_inputs])
    x = tf.split(x,num_steps, 0)

    label = tf.transpose(y_result,[1,0,2])


    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    w_output = tf.Variable(tf.random_normal([num_steps, num_hidden], stddev=0.01, dtype=tf.float32))
    b_output = tf.Variable(tf.random_normal([num_steps, 1], stddev=0.01, dtype=tf.float32))
    the_output = []

    for i in range(num_steps):
        the_output.append(tf.matmul(outputs[i], w_output[i:i+1,:], transpose_b=True)+ b_output[i])

    output = tf.stack(the_output)

 #   return output

#def train_neural_network(x):


  #  prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.pow(output - label,2))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(num_epochs):
            epoch_loss = 0

            epoch_x,epoch_y = gen_data()   # tempX has batch_size elements of shape ( num_steps x num_inputs)
            dict = {x_input: epoch_x, y_result: epoch_y}
            _, c , final_output= sess.run([optimizer, cost, output], feed_dict=dict)
            epoch_loss += c
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)


            #valdict = {inputs: valX, result: valy}
            #updated_state,costval,finaloutput = sess.run((states,cost,output), feed_dict=dict)
            #costval_container.append(costval)
            #print(costval)
        print(finaloutput)


recurrent_neural_network()



# epoch_loss = 0
 #           for _ in range(int(mnist.train.num_examples/batch_size)):
  #              epoch_x, epoch_y = mnist.train.next_batch(batch_size)
   #             epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

    #            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
     #           epoch_loss += c

      #      print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

       # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

#train_neural_network(x)
