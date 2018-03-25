import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.utils import np_utils
import pickle


### Variables ###
data = [[]]
labels = []
testData = [[]]
testLabels = []

### Get data ###
print('Getting data...')
with open('C://Users/riley/Documents/data_batch_1', 'rb') as file:
    dict = pickle.load(file, encoding='bytes')
    data = np.reshape(dict[b'data'], (10000, 3, 32, 32))
    labels = np.asarray(dict[b'labels'])
for i in range(1, 5):
    with open('C://Users/riley/Documents/data_batch_'+str(i+1), 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
        data = np.append(data, np.reshape(dict[b'data'], (10000, 3, 32, 32)), axis = 0)
        labels = np.append(labels, dict[b'labels'])
print((data.shape, labels.shape))
with open('C://Users/riley/Documents/test_batch', 'rb') as file:
    dict = pickle.load(file, encoding='bytes')
    testData = np.reshape(dict[b'data'], (10000, 3, 32, 32))
    testLabels = np.asarray(dict[b'labels'])
### Create model ###
print('Creating model...')
model = Sequential()

model.add(Conv2D(64, kernel_size=(112, 112), strides=7*7/2, activation='relu', input_shape=(3,32,32), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(3*3/2,3*3/2)))
model.add(Conv2D(192, kernel_size=(56, 56), strides = 3*3/1, activation='relu', data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(3*3/2,3*3/2)))
model.add(Inception(Input(shape=(28,28,192))))
model.add(Inception(Input(shape=(28,28,256))))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (3*3/2, 3*3/2)))
model.add(Inception(Input(shape=(14,14,480))))
model.add(Inception(Input(shape=(14,14,512))))
model.add(Inception(Input(shape=(14,14,512))))
model.add(Inception(Input(shape=(14,14,512))))
model.add(Inception(Input(shape=(14,14,528))))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (3*3/2, 3*3/2)))
model.add(Inception(Input(shape=(7,7,832))))
model.add(Inception(Input(shape=(7,7,832))))
model.add(AveragePooling2D(pool_size=(7,7), strides = (7*7/1, 7*7/1)))
model.add(Dropout(0.4))
model.add(Dense(1000, activation='linear'))
model.add(Dense(1000, activation='softmax'))

### Compile model ###
print('Compiling model...')
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

### Fit model ###
print('Fitting model...')
model.fit(data, labels,
          batch_size=32, nb_epoch=10, verbose=1)

### Evaluate model ###
print('Evaluate model...')
score = model.evaluate(testData, testLabels, verbose=0)

### Inception layer ###
def Inception(input):
    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    return keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
