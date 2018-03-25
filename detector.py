import cv2
import pickle
import os
import numpy as np
import pickle
from itertools import chain

### Variables ###
data = [[]]
labels = []
testData = [[]]
testLabels = []
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
print('Getting data...')
with open('C://Users/riley/Documents/data_batch_1', 'rb') as file:
    dict = pickle.load(file, encoding='bytes')
    data = np.reshape(dict[b'data'], (10000, 3, 1024))
    labels = np.asarray(dict[b'labels'])
for i in range(1, 5):
    with open('C://Users/riley/Documents/data_batch_'+str(i+1), 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
        data = np.append(data, np.reshape(dict[b'data'], (10000, 3, 1024)), axis = 0)
        labels = np.append(labels, dict[b'labels'])
print((data.shape, labels.shape))
with open('C://Users/riley/Documents/test_batch', 'rb') as file:
    dict = pickle.load(file, encoding='bytes')
    testData = np.reshape(dict[b'data'], (10000, 3, 1024))
    testLabels = np.asarray(dict[b'labels'])
for i, x in enumerate(labels):
    if x != 4: labels[i] = 0
    else: labels[i] = 1
for i, x in enumerate(testLabels):
    if x != 4: testLabels[i] = 0
    else: testLabels[i] = 1
### Create HOG ###
print('Creating HOG...')
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradients)
print('Computing HOG...')
descriptor = []
for i, im in enumerate(data):
    print(str(i)+'/50000')
    im = im.reshape(3, 32, 32)
    descriptor.append(hog.compute(im[0]))
descriptor = np.squeeze(descriptor)

### SVM ###
print('Creating SVM...')
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR) #RBF might not work on android
svm.setC(12.5)
svm.setGamma(0.50625)
svm.setDegree(3)
print('Training SVM...')
svm.train(descriptor, cv2.ml.ROW_SAMPLE, labels)

print('Saving SVM...')
svm.save('C://Users/riley/Documents/svm_data.dat')

print('Testing SVM...')
descriptorTest = []
for i, im in enumerate(testData):
    print(str(i)+'/10000')
    im = im.reshape(3, 32, 32)
    descriptorTest.append(hog.compute(im[0]))
descriptorTest = np.squeeze(descriptorTest)
testResponse = svm.predict(descriptorTest)[1].ravel()
print('Calculating error...')
diff = 0
for i in range(len(testResponse)):
    if testResponse[i] != testLabels[i]:
        diff += 1
err = diff/len(testResponse)
print(err)
