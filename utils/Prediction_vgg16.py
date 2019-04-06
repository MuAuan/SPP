from __future__ import print_function
import keras
from keras.datasets import cifar10,cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Reshape, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, SGD

import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
import cv2
from getDataSet import getDataSet
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.applications.vgg16 import VGG16

import h5py

num_classes = 100
img_rows=64 #32 #64
img_cols=64 #32 #64
result_dir="./history"

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_test_original = y_test
X_train =[]
X_test = []
for i in range(50000):
    dst = cv2.resize(x_train[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC) #cv2.INTER_LINEAR #cv2.INTER_CUBIC
    dst = dst[:,:,::-1]  #追記；理由はおまけに記載
    X_train.append(dst)
for i in range(10000):
    dst = cv2.resize(x_test[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    dst = dst[:,:,::-1]  #追記；理由はおまけに記載
    X_test.append(dst)
X_train = np.array(X_train)
X_test = np.array(X_test)

y_train=y_train[:50000]
y_test=y_test[:10000]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

x_train = X_train.astype('float32')
x_test = X_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


"""
x_test = []
for i in range(2):
    img = cv2.imread(str(i)+".jpg")
    x_test.append(img)

x_test = np.array(x_test)
print('x_test.shape=',x_test.shape)
X_test = []
dst = cv2.resize(x_test[0], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
dst = dst[:,:,::-1]  #追記；理由はおまけに記載
X_test.append(dst)
X_test = np.array(X_test)

print(X_test.shape)

x_test = X_test.astype('float32')
x_test /= 255
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
#y_test = keras.utils.to_categorical(y_test, num_classes)
"""

# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=x_test.shape[1:]) 
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:])) 
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.75))
top_model.add(Dense(num_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=top_model(vgg16.output))

# Fine-tuningのときはSGDの方がよい⇒adamがよかった
lr = 0.00001
opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# モデルのサマリを表示
model.summary()
model.load_weights('params_model_VGG16_64-64-3_001.hdf5')
plt.imshow(x_test[0])
plt.show()
predict_classes=[]
for i in range(10000):
    answer=model.predict(x_test[i:i+1], batch_size=1, verbose=0, steps=None)
    #print(answer)
    predict_cl = np.argmax(answer)
    predict_classes.append(predict_cl)
    
#print(predict_classes)
#print(answer)
#print(answer.argsort()[0])  #[95:100])

# Prediction
import numpy as np
from sklearn.metrics import confusion_matrix
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)

#predict_classes = model.predict_classes(x_test[1:10,], batch_size=32)
true_classes = y_test_original[0:10000] #np.argmax(y_test[1:10],1)
data = confusion_matrix(true_classes[0:10000], predict_classes[0:10000])
print(data)
import csv
with open('file.csv', 'wt') as f:
    writer = csv.writer(f)
    writer.writerows(data)
    