
from __future__ import print_function
import keras
from keras.datasets import cifar10,cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Reshape, Embedding,InputLayer
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers.noise import GaussianNoise

import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
import cv2
#from getDataSet import getDataSet
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint

import h5py

def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

       
batch_size = 32  #128 #32
num_classes = 10
epochs = 1
data_augmentation = True #True #False
img_rows=128 #112 #96 #160 #128 #32 #64
img_cols=128 #112 #96 #160 #128 #32 #64
result_dir="./history"

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train,y_train,x_test,y_test = getDataSet(img_rows,img_cols)

X_train =[]
X_test = []
for i in range(50000):
    dst = cv2.resize(x_train[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC) #cv2.INTER_LINEAR #cv2.INTER_CUBIC
    dst = dst[:,:,::-1]  
    X_train.append(dst)
for i in range(10000):
    dst = cv2.resize(x_test[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    dst = dst[:,:,::-1]  
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

# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
#input_tensor = Input(shape=x_train.shape[1:]) 
input_tensor = x_train.shape[1:] 
input_model = Sequential()
input_model.add(InputLayer(input_shape=input_tensor))
input_model.add(GaussianNoise(0.01))
input_model.add(UpSampling2D((2, 2)))
 
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_model.output)  #, output=input_model.output)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:])) 
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.75))
top_model.add(Dense(num_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=input_model.input, output=top_model(vgg16.output))

# 最後のconv層の直前までの層をfreeze
#trainingするlayerを指定　VGG16では18,15,10,1など 20で全層固定
for layer in model.layers[1:1]:  
    layer.trainable = False

# Fine-tuningのときはSGDの方がよい⇒adamがよかった
lr = 0.00001
opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# モデルのサマリを表示
model.summary()
#model.load_weights('./cifar10/weights_only_cifar100_160-acc.hdf5')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=1,
                    verbose=1,
                    validation_data=(x_test, y_test))
model.save_weights('./cifar10/weights_cifar10_128U-3_initialG001.hdf5', True) 

for i in range(epochs):
    epoch=200
    if not data_augmentation:
        print('Not using data augmentation.')
        # 学習履歴をプロット
        checkpointer = ModelCheckpoint(filepath='./cifar10/weights_only_cifar10_128UG001-acc.hdf5', 
                                       monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
        checkpointer_acc = ModelCheckpoint(filepath='./cifar10/weights_only_cifar10_128UG001-loss.hdf5', 
                                       monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True)        
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max',
                                       verbose=1)
        lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                                         factor=0.5, min_lr=0.00001, verbose=1)
        csv_logger = CSVLogger('history_cifar10_128UG001_loss_acc.log', separator=',', append=True)
        callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer,checkpointer_acc ]

        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  callbacks=callbacks,          
                  validation_data=(x_test, y_test),
                  shuffle=True)
        
        # save weights every epoch
        model.save_weights('params_model_epoch_karasu_128UG001_{0:03d}.hdf5'.format(0), True)   
        save_history(history, os.path.join(result_dir, 'history_epoch_karasu_128UG001_{0:03d}.txt'.format(i)))
        
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        checkpointer = ModelCheckpoint(filepath='./cifar10/weights_only_cifar10_128UG001-loss.hdf5', 
                                       monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True)
        checkpointer_acc = ModelCheckpoint(filepath='./cifar10/weights_only_cifar10_128UG001-acc.hdf5', 
                                       monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, mode='max',
                                       verbose=1)
        lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                                         factor=0.5, min_lr=0.000001, verbose=1)
        csv_logger = CSVLogger('history_cifar10_128UG001_loss_acc.log', separator=',', append=True) 
        callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer,checkpointer_acc]

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epoch,
                            callbacks=callbacks,          
                            validation_data=(x_test, y_test))
        model.save_weights('params_model_epoch_vgg16_128UG001_a3{0:03d}.hdf5'.format(0), True)   
        save_history(history, os.path.join(result_dir, 'history_epoch_vgg16_128UG001_a3{0:03d}.txt'.format(i)))
