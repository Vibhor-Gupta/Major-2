import readFile
import labels
import cv2

folder="D:\data-1"
path='D:/data-2/'

imlist=[]

for k,v in labels.label.iteritems():
	imlist.append(k)

images = readFile.resize_images(folder,imlist[:100])
i=0

for im in images:
	i+=1
	cv2.imwrite(str(path) + str(i) + '.jpg',im)


