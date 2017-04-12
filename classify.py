import readFile
import cv2
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score
import labels
import numpy as np
import featureGeneration
from sklearn.svm import LinearSVC

folder="D:\data-1"

train=labels.train_image
labels_train=labels.train_labels
test=labels.test_image
labels_test=labels.test_labels

train_images=readFile.resize_images(folder,train)
train_gray_images=readFile.grayscale_images(folder,train)
test_images=readFile.resize_images(folder,test)
test_gray_images=readFile.grayscale_images(folder,test)

# lbp_train_features=featureGeneration.calculate_lbp(train_gray_images)
# lbp_test_features=featureGeneration.calculate_lbp(test_gray_images)

hog_train_features=featureGeneration.calculate_hog(train_images)
hog_test_features=featureGeneration.calculate_hog(test_images)


model=LinearSVC()
model.fit(hog_train_features,labels_train)

res=model.predict(hog_test_features)
acc=accuracy_score(labels_test,res)

print acc