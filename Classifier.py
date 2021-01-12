import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
X,y = fetch_openml('mnist_784',version = 1,return_X_y = True)
xtrain,xtest,ytrain,ytest = train_test_split(X,y,random_state = 9,train_size = 7500,test_size = 2500)
xtrainscale = xtrain/255
xtestscale = xtest/255
clf = LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrainscale,ytrain)
def getprediction(image):
        Image_PIL = Image.open(image)
        image_bw = Image_PIL.convert('L')
        image_bw_resize = image_bw.resize((28,28),Image.ANTIALIAS)
       #image_bw_resize_inverter = PIL.ImageOps.invert(image_bw_resize)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resize,pixel_filter)
        image_bw_resize_inverter_scale = np.clip(image_bw_resize-min_pixel,0,255)
        max_pixel = np.max(image_bw_resize)
        image_bw_resize_inverter_scale = np.asarray(image_bw_resize_inverter_scale)/max_pixel
        test_sample = np.array(image_bw_resize_inverter_scale).reshape(1,784)
        test_prediction = clf.predict(test_sample)
        return test_prediction[0]