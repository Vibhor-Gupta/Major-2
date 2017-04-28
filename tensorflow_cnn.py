import readFile
# import cv2
# from skimage.feature import local_binary_pattern
# from sklearn.metrics import accuracy_score
import labels
import numpy as np
# import featureGeneration
# from sklearn.svm import LinearSVC
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
# from tensorflow.contrib.learn import ModelFnOps

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model(features,labels,mode):
  in_layer=tf.reshape(features,[-1,128,64,3])

  conv1 = tf.layers.conv2d(inputs=in_layer,filters=8,kernel_size=[5, 5],padding="valid",activation=tf.nn.relu)

  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=[5, 3],padding="valid",activation=tf.nn.relu)

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  conv3 = tf.layers.conv2d(inputs=pool2,filters=32,kernel_size=[6, 3],padding="valid",activation=tf.nn.relu)

  conv4 = tf.layers.conv2d(inputs=conv3,filters=64,kernel_size=[5, 3],padding="valid",activation=tf.nn.relu)

  conv5 = tf.layers.conv2d(inputs=conv4,filters=128,kernel_size=[5, 3],padding="valid",activation=tf.nn.relu)

  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

  pool5_flat = tf.reshape(pool2, [-1, 8 * 4 * 128])
  dense6 = tf.layers.dense(inputs=pool5_flat, units=1024, activation=tf.nn.relu)
  dropout6 = tf.layers.dropout(inputs=dense6, rate=0.5, training=mode == learn.ModeKeys.TRAIN)

  dense7 = tf.layers.dense(inputs=dropout6, units=1024, activation=tf.nn.relu)
  dropout7 = tf.layers.dropout(inputs=dense7, rate=0.5, training=mode == learn.ModeKeys.TRAIN)

  output = tf.layers.dense(inputs=dropout7, units=7)

  loss=0
  train_op=0

  if mode != learn.ModeKeys.INFER:
      onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=7)
      loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, output=output)

  if mode == learn.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss(loss=loss,global_step=tf.contrib.framework.get_global_step(),learning_rate=0.01,optimizer="SGD")

  predictions = {"classes": tf.argmax(input=output, axis=1),"probabilities": tf.nn.softmax(output, name="softmax_tensor")}

  return model_fn_lib.ModelFnOps(mode=mode,predictions=predictions,loss=loss,train_op=train_op)

folder="D:\data-1"

train=labels.train_image
labels_train=labels.train_labels
test=labels.test_image
labels_test=labels.test_labels

train_images=readFile.resize_images(folder,train)
test_images=readFile.resize_images(folder,test)

train_im=np.array(train_images).reshape(-1,128,64,3)
test_im=np.array(test_images).reshape(-1,128,64,3)

training_label=np.array(labels_train)
testing_label=np.array(labels_test)

food_classifier = learn.Estimator(model_fn=cnn_model, model_dir="/tmp/convnet_model")

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=2)

food_classifier.fit(x=train_im,y=training_label,batch_size=10,steps=20,monitors=[logging_hook])

metrics = {"accuracy":learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),}

eval_results = food_classifier.evaluate(x=test_im, y=testing_label, metrics=metrics)
print(eval_results)