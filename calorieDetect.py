import readFile
import cv2
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import labels
import numpy as np
import featureGeneration
from sklearn.svm import LinearSVC
import os
from sklearn import  linear_model
from sklearn.svm import SVR

folder="D:\data-1"

train=labels.train_image
labels_train=labels.regression_train
test=labels.test_image
labels_test=labels.regression_test

train_images=readFile.resize_images(folder,train)
train_gray_images=readFile.grayscale_images(folder,train)
test_images=readFile.resize_images(folder,test)
test_gray_images=readFile.grayscale_images(folder,test)


# lbp_train_features=featureGeneration.calculate_lbp(train_gray_images)
# lbp_test_features=featureGeneration.calculate_lbp(test_gray_images)
# lbp_train=np.array(lbp_train_features)
# lbp_test=np.array(lbp_test_features)

# print lbp_test.shape
hog_train_features=featureGeneration.calculate_hog(train_images)
hog_test_features=featureGeneration.calculate_hog(test_images)
hog_train = np.array(hog_train_features).reshape(75,3780)
hog_test = np.array(hog_test_features).reshape(25,3780)

svr_rbf = SVR(kernel='poly', C=1e3, gamma=0.1)
# svr_lin = SVR(kernel='linear', C=1e3)
# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
res = svr_rbf.fit(hog_train, labels_train).predict(hog_test)
# y_lin = svr_lin.fit(X, y).predict(X)
# y_poly = svr_poly.fit(X, y).predict(X)


# reg = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
# reg.fit(hog_train,labels_train)

# print hog_train
# print labels_train
# res=reg.predict(hog_test)
print res,labels_test

s=0
for i in range(0,len(res)):
	s+= (abs(res[i]-labels_test[i])/labels_test[i])

acc=mean_absolute_error(labels_test,res)
acc1=mean_squared_error(labels_test,res)

print acc
print acc1
print s/len(res)

