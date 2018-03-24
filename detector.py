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
        data.append(dict[b'data'])
        labels.append(dict[b'labels'])


### Create HOG ###
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradients)
descriptor = []
# for im in data:
#     descriptor.append(hog.compute(im))
descriptor.append(hog.compute(data[0]))

### SVM ###
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF) #RBF might not work on android
svm.setC(12.5)
svm.setGamma(0.50625)
svm.trainAuto(descriptor, cv2.ml.ROW_SAMPLE, labels)
