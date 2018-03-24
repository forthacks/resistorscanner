import cv2
import pickle
import os
import numpy as np
import pickle

### Variables ###
data = []
labels = []
winSize = (32,32)
blockSize = (8,8)
blockStride = (4,4)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True

### Get data ###
for i in range(5):
    with open('C://Users/riley/Documents/data_batch_'+str(i+1), 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
        data.append(dict['data'])
        labels.append(dict['labels'])


### Create HOG ###
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradients)


### SVM ###
svm = cv2.ml.SVM_create()
svm.setType(cv2.m1.SVM_C_SVC)
scm.setKernel(cv2.ml.SVM_RBF) #RBF might not work on android
svm.setC(C)
svm.setGamma(gamma)
svm.train(data, cv2.ml.ROW_SAMPLE, labels)
