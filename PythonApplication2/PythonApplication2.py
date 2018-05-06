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
    #layer 모듈에서 입력은 [batch_size, image_height, image_width, channels] 를 기본 인수로 필요로함
    #batch_size:훈련에서 사용할 subset 수, image_height/width:이미지 픽셀크기, channels:흑백(1,Black),컬러(3,RGB)
    #이 튜토리얼에서 28x28 크기의 흑백 이미지가 input이므로 [batch_size,28,28,1] 임 
    #reshape 메소드를 사용해서 훈련용 subset정보를 value로 가지는 features를 원하는 형태로 변환할 수 있음
    #-1 값은 reshape를 할 때 동적으로 결정하라는 의미
    #예를 들어 features가 28*28*1개 짜리 배열이라면 batch_size는 자동으로 1이됨
    input_layer = tf.reshape(features["x"],[-1,28,28,1])
    
    #Convolutional Layer #1
    #input은 반드시 [batch_size,image_height,image_width,channels] 형태여야 함
    #filters는 필터의 갯수, kernel_size는 합성곱타일의 크기 5x5 처럼 가로,세로가 같으면 그냥 5해도 됨
    #padding="same"이라고 하면 합성곱 후 생기는 24x24 크기의 출력에 0패딩을 붙여서 input과 같은 크기의 출력을 내보냄
    #activation은 활성화 함수로 예전까지는 sigmoid 함수를 사용했으나 ReLU(Rectified Linear Unit)함수가 훨씬 좋은 예측을 낸다
    #conv2d 를 수행하고 나면 [batch_size,28,28,1]형태의 결과가 나온다
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    #Pooling Layer #1
    #input은 반드시 [batch_size,image_height,image_width,channels] 형태여야 함
    #pool_size는 max pooling을 할 필터의 크기
    #strides 는 필터링 후 다음 픽셀간의 보폭을 얼마나 할지로 가로세로가 같으면 2 다를 경우는 [x,y]로 쓰면 된다
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
    #dense layer를 사용하기 전에 마지막 pooling결과인 pool2를 flattening 작업을 해줘야 함=dense layer 입력에 맞게 형태 변환
    #pool2는 [batch_size,7,7,64] 크기임
    #다시 reshape 함수를 써서 pool2에 있는 value들을 [batch_size,7*7*64] 크기로 변환함
    #그 후에 dense layer의 입력으로 넣을 수 있고 설계에 따라 1024개의 유닛을 지정하고 활성화함수는 ReLU를 쓴다.
    #dropout은 너무 많은 데이터를 줄이기 위한 것으로 0.4로 설정함에 따라 임의로 40%는 제거한다.
    #훈련 과정일 때만 dropout을 적용하기 위해 boolean 값인 tf.estimator.ModeKeys.TRAIN 을 사용한다.
    #dropout 크기는 [batch_size,1024]가 나온다
    pool2_flat=tf.reshape(pool2,[-1,7*7*64])
    dense=tf.layers.dense(input=pool2_flat,units=1024,activation=tf.nn.relu)
    dropout=tf.layers.dropout(inputs=dense,rate=0.4,training=mode == tf.estimator.ModeKeys.TRAIN)
    #Logits Layer
    #마지막 Logits layer에서는 dropout을 인풋으로 각각 0~9를 의미하는 10개의 유닛을 가진 dense layer를 사용
    #최종 산출물은 [batch_size,10] 크기
    logits= tf.layers.dense(inputs=dropout, units=10)
    predictions= {
        #generate predictions (for PREDICt and EVAL mode)
        "classes":tf.arg_max(input=logits,axis=1),
        #Add 'softmax_tensor' to the graph. It is used for PREDICT and by the 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    #Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,llogits=logits)
    #Configure the Training Op(for TRAIN mode)
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op=optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #Add evaluation metrics(for EVAL mode)
    eval_metric_ops={
        "accuracy":tf.metrics.accuracy(
            labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    #Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels=np.asarray(minist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels=np.asarray(mnist.test.labels, dtype=np.int32)

    #Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    #Set up logging for predictions
    tensors_to_log = {"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors = tensors_to_log, every_n_iter=50)

    #Train the model
    train_input_fn = tfilter.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        train_input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    #Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)