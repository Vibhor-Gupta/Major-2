import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import readFile
import cv2
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score
import labels
import featureGeneration
from sklearn.svm import LinearSVC


folder="D:\data-1"

train=labels.train_image
labels_train=labels.train_labels
test=labels.test_image
labels_test=labels.test_labels

train_images=readFile.resize_images(folder,train)
test_images=readFile.resize_images(folder,test)

train_im=np.array(train_images).reshape(-1,3,128,64)
test_im=np.array(test_images).reshape(-1,3,128,64)

training_label=np.array(labels_train).astype(np.uint8)
testing_label=np.array(labels_test).astype(np.uint8)

convolutionNet=NeuralNet(
	layers=[('input',layers.InputLayer),
			('conv2d1',layers.Conv2DLayer),
			('maxpool1',layers.MaxPool2DLayer),
			('conv2d2',layers.Conv2DLayer),
			('maxpool2',layers.MaxPool2DLayer),
			('conv2d3',layers.Conv2DLayer),
			('conv2d4',layers.Conv2DLayer),
			('conv2d5',layers.Conv2DLayer),
			('maxpool5',layers.MaxPool2DLayer),
			('dense6',layers.DenseLayer),
			('dropout6',layers.DropoutLayer),
			('dense7',layers.DenseLayer),
			('dropout7',layers.DropoutLayer),
			('output', layers.DenseLayer)
		],

	#input_layer
	input_shape=(None,3,128,64),

	#conv2d1
	conv2d1_num_filters=8,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),

    #maxpool1
    maxpool1_pool_size=(2, 2),

    #conv2d2
    conv2d2_num_filters=16,
    conv2d2_filter_size=(5, 3),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
	
	#maxpool2
	maxpool2_pool_size=(2, 2),

	#conv2d3
    conv2d3_num_filters=32,
    conv2d3_filter_size=(6, 3),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,

    #conv2d4
    conv2d4_num_filters=64,
    conv2d4_filter_size=(5, 3),
    conv2d4_nonlinearity=lasagne.nonlinearities.rectify,

    #conv2d5
    conv2d2_num_filters=128,
    conv2d2_filter_size=(5, 3),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,

    #dense6
    dense6_num_units=4096,
    dense6_nonlinearity=lasagne.nonlinearities.rectify,

    #drop6
    dropout6_p=0.5,

	#dense7
    dense7_num_units=4096,
    dense7_nonlinearity=lasagne.nonlinearities.rectify,    

    #drop7
    dropout7_p=0.5,

    #output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=1024,

    #optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
	)

nn=convolutionNet.fit(train_im,training_label)
preds=convolutionNet.predict(test_im)

acc=accuracy_score(testing_label,preds)
print acc

# #feature extraction from Neural Net


# train_2=labels.train_image_2
# labels_train_2=labels.train_labels_2
# test_2=labels.test_image_2
# labels_test_2=labels.test_labels_2

# train_images_2=readFile.resize_images(folder,train_2)
# test_images_2=readFile.resize_images(folder,test_2)

# train_im_2=np.array(train_images_2).reshape(-1,3,128,64)
# test_im_2=np.array(test_images_2).reshape(-1,3,128,64)

# training_label_2=np.array(labels_train_2).astype(np.uint8)
# testing_label_2=np.array(labels_test_2).astype(np.uint8)

# dense_layer6 = layers.get_output(convolutionNet.layers_['dense6'], deterministic=True)
# dense_layer7 = layers.get_output(convolutionNet.layers_['dense7'], deterministic=True)
# input_var = convolutionNet.layers_['input'].input_var

# f_dense6=theano.function([input_var], dense_layer6)
# f_dense7=theano.function([input_var], dense_layer7)

# #train svm on dense6 features
# dense6_features=[]
# dense6_features_test=[]

# for im in train_im_2:
#     pred=f_dense6(im)
#     dense6_features.append(pred.ravel())

# for im in test_im_2:
#     pred=dense6(im)
#     dense6_features_test.append(pred.ravel())

# model=LinearSVC()
# model.fit(dense6_features,training_label_2)

# res=model.predict(dense6_features_test)
# acc1=accuracy_score(testing_label_2,res)

# print acc1

# #train svm on dense7 features
# dense7_features=[]
# dense7_features_test=[]

# for im in train_im_2:
#     pred=f_dense7(im)
#     dense7_features.append(pred.ravel())

# for im in test_im_2:
#     pred=dense7(im)
#     dense7_features_test.append(pred.ravel())

# mode1=LinearSVC()
# mode1.fit(dense7_features,training_label_2)

# res=mode1.predict(dense7_features_test)
# acc2=accuracy_score(testing_label_2,res)

# print acc2
