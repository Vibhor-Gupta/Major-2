import readFile
import labels
import cv2
import theano
import os

folder="D:\data-1"
# path='D:/data-2/'

# imlist=[]

# for k,v in labels.label.iteritems():
# 	imlist.append(k)

# images = readFile.resize_images(folder,imlist[:100])
# i=0

# for im in images:
# 	i+=1
# 	cv2.imwrite(str(path) + str(i) + '.jpg',im)

# print theano.config.floatX
item="1.jpg"
img = cv2.imread(os.path.join(folder,item))
img1=img[1000:1944,300:2400]
img2=cv2.resize(img1,(128,64))
cv2.putText(img2,"Hello",(0,20),cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
cv2.imshow("Image",img2)
cv2.waitKey(0)