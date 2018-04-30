# Inspired by the TF Tutorial: https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
import numpy as np
from data_gen import Gymmer

# Mnist Data
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Constants to eventually parameterise
## Base Dir to write logs to
BASE_LOGDIR = './logs/'
## Subdirectory for this experiment
#RUN = 'SpaceInvaders-v0'
#RUN = 'Pong-v0'
#RUN = 'Assault-v0'
#RUN = 'Atlantis-v0'
#RUN = 'AirRaid-v0'
RUN = 'Alien-v0'
## Learning Rate for Adam Optimizer
LEARN_RATE = 1e-4
## Number of images to push through the network at a time
#BATCH_SIZE = 256 
BATCH_SIZE = 16 
#BATCH_SIZE = 64 
## Number of Epochs to train for
MAX_EPOCHS = 10
## How many training steps between outputs to screen and tensorboard
output_steps = 2 
## Enable or disable GPU (0 disables GPU, 1 enables GPU)
SESS_CONFIG = tf.ConfigProto(device_count = {'GPU': 1})

# Define functions that create useful variables
def weight_variable(shape, name="W"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="B"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 2D Convolution Func 
def conv2d(x, W, name='conv'):
    with tf.name_scope(name):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max-Pooling Function - Pooling explained here: 
# http://ufldl.stanford.edu/tutorial/supervised/Pooling/
def max_pool_2x2(x, name='max_pool'):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Define a Convolutional Layer 
def conv_layer(x, fan_in, fan_out, name="convl"):
    with tf.name_scope(name):
        # Create Weight Variables
        W = weight_variable([5, 5, fan_in, fan_out], name="W")
        B = bias_variable([fan_out], name="B")
        # Convolve the input using the weights
        conv = conv2d(x, W)
        # Push input+bias through activation function
        activ = tf.nn.relu(conv + B)
        # Create histograms for visualization
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Biases", B)
        tf.summary.histogram("Activations", activ) 
        # MaxPool Output
        return max_pool_2x2(activ)


data = Gymmer(RUN)
# Get Data
MAX_TRAIN_STEPS = int(MAX_EPOCHS*data.max_steps/BATCH_SIZE)
SIZE_X = 210     # Number of pixels in x direction
SIZE_Y = 160     # Number of pixels in y direction
SIZE_C = 3      # Number of pixels in c

# Begin Defining the Computational Graph
with tf.name_scope('MainGraph'):
    with tf.name_scope('Inputs'):
        # Placeholders for data and labels
        ## Mnist gives images as flat vectors, thus the size [None, SizeX*SizeY] 
        ## instead of the more intuitive [None, SizeX, SizeY]
        x = tf.placeholder(tf.float32, shape=[None, SIZE_X, SIZE_Y, SIZE_C])

        ## Ground-Truth Labels, as 1-hot vectors
        y_true = tf.placeholder(tf.float32, shape=[None])

        ## Dropout probability. Dropout is similar to model averaging
        ## https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        keep_prob = tf.placeholder(tf.float32)

        # Reshape X to make it into a 2D image
        x_image = tf.reshape(x, [-1, SIZE_X, SIZE_Y, 3])
        tf.summary.image('sample_image', x_image, 3)

    # Convolution Layers
    conv1 = conv_layer(x_image, 3, 32, name='Conv1') 
    conv2 = conv_layer(conv1, 32, 64, name='Conv2') 
    
    # Create image summaries to visualize layer outputs in Tensorboard
    tf.summary.image('conv1_viz', tf.expand_dims(conv1[:,:,:,1], axis=3), 3)
    tf.summary.image('conv2_viz', tf.expand_dims(conv2[:,:,:,1], axis=3), 3)

    # Fully Connected Layers
    with tf.name_scope('FC1'):
        W_fc1 = weight_variable([53*40*64, 512])
        b_fc1 = bias_variable([512])
        
        h_pool2_flat = tf.reshape(conv2, [-1, 53*40*64])
        #h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1 = tf.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Split into two
        phi_l = h_fc1[0::2]
        phi_r = h_fc1[1::2]

    with tf.name_scope('FC2'):
        W_fc2 = weight_variable([1024, 1024])
        b_fc2 = bias_variable([1024])
        ## Dropout Layer
        #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        ## FC Layer 2 - Output Layer
        #y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        phi_conc = tf.concat([phi_l, phi_r], 1)
        h_fc2 = tf.sigmoid(tf.matmul(phi_conc, W_fc2) + b_fc2)

    with tf.name_scope('FC3'):
        W_fc3 = weight_variable([1024, 1])
        b_fc3 = bias_variable([1])
        y_pred = tf.matmul(h_fc2, W_fc3) + b_fc3

    with tf.name_scope('Objective'):
        # Define the objective function
        mse = tf.losses.mean_squared_error(y_true, tf.squeeze(y_pred))
        tf.summary.scalar('mse', mse)


# Define the training step
train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(mse)

# Create the session
sess = tf.Session(config=SESS_CONFIG)

# Init all weights
sess.run(tf.global_variables_initializer())

# Merge Summaries and Create Summary Writer for TB
all_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(BASE_LOGDIR + RUN)
writer.add_graph(sess.graph) 


# Train 
with sess.as_default():
    for cur_step in range(MAX_TRAIN_STEPS):
        batch, labels = data.get_batch(BATCH_SIZE)
        if cur_step % output_steps == 0:
            out_mse, predictions = sess.run([mse, y_pred], feed_dict={x: batch, y_true: labels, keep_prob: 1.0})
            print('Step: ' + str(cur_step) + '\t\tTrain MSE: ' + str(round(out_mse, 2)))
            #print(predictions[::1]*data.max_steps)
            #print(labels[::1]*data.max_steps)
            #print("Avg_Error", str(np.transpose(predictions[::1]*data.max_steps) - labels[::1]*data.max_steps))
            print("Avg_Error", str((abs(np.transpose(predictions)*data.max_steps - labels*data.max_steps)).mean()))
            # Calculate and write-out all summaries
            # Validate on batch from validation set
            all_sums = sess.run(all_summaries, feed_dict={x: batch, y_true: labels, keep_prob: 1.0})
            writer.add_summary(all_sums, cur_step) 
        train_step.run(feed_dict={x: batch, y_true: labels, keep_prob: 0.5})

    # Done training, now show final results
    sum_err = 0
    for i in range(10):
        data.refill_bag()
        batch, labels = data.get_batch(BATCH_SIZE)
        out_mse, predictions = sess.run([mse, y_pred], feed_dict={x: batch, y_true: labels, keep_prob: 1.0})
        sum_err += (abs(np.transpose(predictions)*data.max_steps - labels*data.max_steps)).mean()
    print("Avg Error")
    print(sum_err/10)

