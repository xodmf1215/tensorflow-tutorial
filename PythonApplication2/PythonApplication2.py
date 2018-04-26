from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__" :
    tf.app.run()

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    #input Layer
    input_layer = tf.reshape(features["x"],[-1,28,28,1])
    
    #Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    #Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2)
    #Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    #Dense Layer
    pool2_flat=tf.reshape(pool2,[-1,7*7*64])
    dense=tf.layers.dense(input=pool2_flat,units=1024,activation=tf.nn.relu)
    dropout=tf.layers.dropout(inputs=dense,rate=0.4,training=mode == tf.estimator.ModeKeys.TRAIN)
    #Logits Layer
    logits= tf.layers.dense(inputs=dropout, units=10)
    predictions= {
        #generate predictions (for PREDICt and EVAL mode)
        "classes":tf.arg_max(input=logits,axis=1),
        #Add 'softmax_tensor' to the graph. It is used for PREDICT and by the 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys