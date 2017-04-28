import cv2
import os
import labels
# from sklearn.cross_validation import train_test_split

def load_images_from_folder(folder,data):
    images = []
    for item in data:
        img = cv2.imread(os.path.join(folder,item))
        if img is not None:
            images.append(img)
    return images

def crop_images(folder,data):
 	images=load_images_from_folder(folder,data)
 	cropped=[]
 	for im in images:
 		crop=im[1000:1944,300:2400]
 		cropped.append(crop)
 	return cropped

def resize_images(folder,data):
    images=crop_images(folder,data)
    resized=[]
    for im in images:
        re=cv2.resize(im,(128,64))
        resized.append(re)
    return resized

def grayscale_images(folder,data):
    images=resize_images(folder,data)
    gray=[]
    for im in images:
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray.append(gray_im)
    return gray
