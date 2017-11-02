# First shot a a NN
# requires load_iterators.py, which depends on files preprocessing_train_data.py produces
# On my computer, running on CPUs, this takes ~3 seconds per run through the loop

import tensorflow as tf
import numpy as np
from time import time

from load_iterators import *

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def max_pool_nxn(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1],
                        strides=[1, n, n, 1], padding='SAME')
                        
train_gen, val_gen = load_iterators()

print("Loaded iterators")

n_cats = 5270

#Dimensions of NN
conv_1_size = 4
conv_1_channels = 16
conv_1_pool = 2

conv_2_size = 4
conv_2_channels = 32
conv_2_pool = 3

conv_3_size = 4
conv_3_channels= 64
conv_3_pool = 2

fully_connected_1_size = 8096

print_every = 100

unpacked_size = conv_3_channels * 180 * 180 / (conv_1_pool * conv_2_pool * conv_3_pool)**2
print(unpacked_size)
if not (unpacked_size % 1) == 0:
    raise ValueError("Pooling size error: needs to divide 180 into integer")
unpacked_size = int(unpacked_size)

#set up placeholders
x = tf.placeholder(tf.float32, shape=[None, 180, 180, 3], name='input_images')
y_ = tf.placeholder(tf.float32, shape=[None, n_cats], name='one_hotted_labels')

#set up neurons for 1st convolutional layer
W_conv1 = weight_variable([conv_1_size, conv_1_size, 3, conv_1_channels])
b_conv1 = bias_variable([conv_1_channels])

#don't know why this step is necessary, but it is
x_image = tf.reshape(x, [-1,180,180,3])

#do the first convolutional layer, pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_nxn(h_conv1, conv_1_pool)

#do the second convolutional layer, pooling
W_conv2 = weight_variable([conv_2_size, conv_2_size, conv_1_channels, conv_2_channels]) #4x4 templates, 32 input channels, 64 output channels
b_conv2 = bias_variable([conv_2_channels])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_nxn(h_conv2, conv_2_pool)

#do the second convolutional layer, pooling
W_conv3 = weight_variable([conv_3_size, conv_3_size, conv_2_channels, conv_3_channels]) #4x4 templates, 32 input channels, 64 output channels
b_conv3 = bias_variable([conv_3_channels])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_nxn(h_conv3, conv_3_pool)

h_pool3_flat = tf.reshape(h_pool3, [-1, unpacked_size])

#Densely connected layer
W_fc1 = weight_variable([unpacked_size, fully_connected_1_size])
b_fc1 = bias_variable([fully_connected_1_size])

h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#readout layer
W = weight_variable([fully_connected_1_size, n_cats])
b = bias_variable([n_cats])

y_conv = tf.matmul(h_fc1_drop, W) + b

sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))


train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1, name='argmax_prediction'), tf.argmax(y_,1, name='argmax_actual_val'))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

running_tot = np.zeros(5270, )

print("Starting training")
#Do the training
for i in range(15000):
    bx, by = next(train_gen)
    
    running_tot += np.sum(by, 0)
    
    if i%10 == print_every:
        #print(i)
        init_acc = accuracy.eval(feed_dict={x: bx, y_: by, keep_prob: 1.0})
        starttime = time()
    
    train_step.run(feed_dict={x: bx, y_: by, keep_prob: 0.5})
    
    if i%10 == print_every:
        elapsed = time() - starttime
        #print('After training on those samples:')
        aft_acc = accuracy.eval(feed_dict={x: bx, y_: by, keep_prob: 1.0})
        print("after {} cycles, input batch went from {} correct to {:.2f}, took {:.2f} seconds".format(i, init_acc*128, aft_acc*128, elapsed))
        print("Median; mean number of images for each category is {}; {}".format(np.median(running_tot), np.mean(running_tot)))
        
print("Saving model...")
saver = tf.train.Saver()
save_path = saver.save(sess, "saved_trained_model.ckpt")
