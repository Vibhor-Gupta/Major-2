import readFile
import cv2
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score
import labels
import numpy as np
import featureGeneration
from sklearn.svm import LinearSVC
import os

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
# lbp_train=np.array(lbp_train_features)
# lbp_test=np.array(lbp_test_features)

# print lbp_test.shape
hog_train_features=featureGeneration.calculate_hog(train_images)
hog_test_features=featureGeneration.calculate_hog(test_images)
hog_train = np.array(hog_train_features).reshape(75,3780)
hog_test = np.array(hog_test_features).reshape(25,3780)

model=LinearSVC()
model.fit(hog_train,labels_train)

res=model.predict(hog_test)
acc=accuracy_score(labels_test,res)

print acc

item="3.jpg"
img = cv2.imread(os.path.join(folder,item))
img3=img[1000:1944,300:2400]
img4=cv2.resize(img3,(600,480))

img1=readFile.grayscale_images(folder,[item])

hog_show_test=featureGeneration.calculate_hog(img1)
hog_show=np.array(hog_show_test).reshape(1,3780)

pred=model.predict(hog_show)[0]

text=""

if(pred==0):
	text="Breakfast Sandwhich"
if(pred==1):
	text="Toast Sandwhich"
if(pred==2):
	text="Donut"
if(pred==3):
	text="Pizza"
if(pred==4):
	text="Salad"
if(pred==5):
	text="Chicken Nugget"
if(pred==6):
	text="Burger"

cv2.putText(img4,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
cv2.imshow("Image",img4)
cv2.waitKey(0)