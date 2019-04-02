
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Reshape, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt

from getDataSet import getDataSet

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


batch_size = 32
num_classes = 3
epochs = 10
data_augmentation = False #True #False
img_rows=128
img_cols=128
result_dir="./history"

# The data, shuffled and split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train,y_train,x_test,y_test = getDataSet(img_rows,img_cols)

x_train = np.array(x_train)  #/ 255
y_train = np.array(y_train).astype(np.int32)
x_test = np.array(x_test) #/ 255
y_test = np.array(y_test).astype(np.int32)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=x_train.shape[1:]) 
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

from keras.models import Model
layer_name1 = 'block3_pool'
intermediate_layer_model = Model(inputs=vgg16.input,
                                 outputs=vgg16.get_layer(layer_name1).output)

# FC層を構築
inputs0 = Input(shape=intermediate_layer_model.output_shape[1:])
# Block 4
block4_conv1 = Conv2D(512, (3, 3),name='block4_conv1', activation='relu', padding='same')(inputs0)
block4_conv2 = Conv2D(512, (3, 3),name='block4_conv2', activation='relu', padding='same')(block4_conv1)
block4_conv3 = Conv2D(512, (3, 3),name='block4_conv3', activation='relu', padding='same')(block4_conv2)
#bn4 = BatchNormalization(axis=3)(conv4_3)
block4_pool = MaxPooling2D(pool_size=(2, 2),name='block4_pool')(block4_conv3)
#drop4 = Dropout(0.5)(pool4)
block4_model = Model(inputs=inputs0, outputs=block4_pool)

inputs1 = Input(shape=block4_model.output_shape[1:])
# Block 5
block5_conv1 = Conv2D(512, (3, 3),name='block5_conv1', activation='relu', padding='same')(inputs1)
block5_conv2 = Conv2D(512, (3, 3),name='block5_conv2', activation='relu', padding='same')(block5_conv1)
block5_conv3 = Conv2D(512, (3, 3),name='block5_conv3', activation='relu', padding='same')(block5_conv2)
#bn5 = BatchNormalization(axis=3)(conv5_3)
block5_pool = MaxPooling2D(pool_size=(2, 2),name='block5_pool')(block5_conv3)
#drop5 = Dropout(0.5)(pool5)
block5_model = Model(inputs=inputs1, outputs=block5_pool)

inputs2 = Input(shape=block5_model.output_shape[1:])
# top_model
x = Flatten()(inputs2)
x = Dense(15*num_classes, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation="softmax")(x)

# topは別途パラメータを読み込みたいので
# topだけで一度Modelを作る
top_model = Model(inputs=inputs2, outputs=predictions)
model = Model(inputs=intermediate_layer_model.input, outputs=top_model(block5_model(block4_model(intermediate_layer_model.output))))

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
print('model.summary()')
model.summary()
print('intermediate_layer_model.summary()')
intermediate_layer_model.summary()
print('block4_model.summary()')
block4_model.summary()
print('block5_model.summary()')
block5_model.summary()
print('top_model.summary()')
top_model.summary()
intermediate_layer_model.save_weights('params_intermediate_model'+layer_name1+'.hdf5', True) 
top_model.save_weights('params_initial_top_model'+layer_name1+'.hdf5', True) 
model.save_weights('params_initial_model'+layer_name1+'.hdf5', True) 
#model.load_weights('params_model_VGG16L3_i_190.hdf5')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

for i in range(epochs):
    epoch=10
    if not data_augmentation:
        print('Not using data augmentation.')
        """
        history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epoch,
                    verbose=1,
                    validation_split=0.1)
        """
        # 学習履歴をプロット
        #plot_history(history, result_dir)
        
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_data=(x_test, y_test),
                  shuffle=True)
        
        # save weights every epoch
        model.save_weights('params_model_epoch_karasu_50_{0:03d}'.format(i)+layer_name1+'.hdf5', True)   
        save_history(history, os.path.join(result_dir, 'history_epoch_karasu_50'+layer_name1+'_{0:03d}.txt'.format(i)))
        intermediate_layer_model.save_weights('params_intermediate_model_{0:03d}'.format(i)+layer_name1+'.hdf5', True) 
        top_model.save_weights('params_initial_top_model_{0:03d}'.format(i)+layer_name1+'.hdf5', True) 
        
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

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epoch,
                            validation_data=(x_test, y_test))
        model.save_weights('params_model_epoch_karasu_a2{0:03d}.hdf5'.format(i), True)   
        save_history(history, os.path.join(result_dir, 'history_epoch_karasu_a2{0:03d}.txt'.format(i)))

    if i%10==0:
        print('i, ir= ',i, lr)
        # save weights every epoch
        model.save_weights('params_model_karasu_VGG16L3_i_3{0:03d}.hdf5'.format(i), True)
        
        lr=lr*0.8
        opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6)
        
        # Let's train the model using Adam
        model.compile(loss='categorical_crossentropy',
                  optimizer=opt,metrics=['accuracy'])
    else:
        continue
model.save_weights('params_model_karasu_VGG16L3_i_'+layer_name1+'{0:03d}.hdf5'.format(epochs), True)
save_history(history, os.path.join(result_dir, 'history_karasu_50'+layer_name1+'.txt'))


"""
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 128, 128, 3)       0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 128, 128, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 128, 128, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 64, 64, 64)        0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 64, 64, 128)       73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 64, 64, 128)       147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 32, 32, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 32, 32, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 32, 32, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 32, 32, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 16, 16, 256)       0
_________________________________________________________________
model_2 (Model)              (None, 8, 8, 512)         5899776
_________________________________________________________________
model_3 (Model)              (None, 4, 4, 512)         7079424
_________________________________________________________________
model_4 (Model)              (None, 3)                 368823
=================================================================
Total params: 15,083,511
Trainable params: 15,083,511
Non-trainable params: 0
_________________________________________________________________
intermediate_layer_model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 128, 128, 3)       0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 128, 128, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 128, 128, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 64, 64, 64)        0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 64, 64, 128)       73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 64, 64, 128)       147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 32, 32, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 32, 32, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 32, 32, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 32, 32, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 16, 16, 256)       0
=================================================================
Total params: 1,735,488
Trainable params: 1,735,488
Non-trainable params: 0
_________________________________________________________________
block4_model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         (None, 16, 16, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 16, 16, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 16, 16, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 16, 16, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 8, 8, 512)         0
=================================================================
Total params: 5,899,776
Trainable params: 5,899,776
Non-trainable params: 0
_________________________________________________________________
block5_model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_3 (InputLayer)         (None, 8, 8, 512)         0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 8, 8, 512)         2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 8, 8, 512)         2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 8, 8, 512)         2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0
=================================================================
Total params: 7,079,424
Trainable params: 7,079,424
Non-trainable params: 0
_________________________________________________________________
top_model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_4 (InputLayer)         (None, 4, 4, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 8192)              0
_________________________________________________________________
dense_1 (Dense)              (None, 45)                368685
_________________________________________________________________
dropout_1 (Dropout)          (None, 45)                0
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 138
=================================================================
Total params: 368,823
Trainable params: 368,823
Non-trainable params: 0
_________________________________________________________________
Train on 170 samples, validate on 59 samples
Epoch 7/10
170/170 [==============================] - 1s 8ms/step - loss: 0.1337 - acc: 0.9529 - val_loss: 0.5312 - val_acc: 0.8475
Epoch 8/10
170/170 [==============================] - 1s 8ms/step - loss: 0.1697 - acc: 0.9235 - val_loss: 0.3897 - val_acc: 0.8983
Epoch 9/10
170/170 [==============================] - 1s 8ms/step - loss: 0.1005 - acc: 0.9588 - val_loss: 0.3644 - val_acc: 0.9153
Epoch 10/10
170/170 [==============================] - 1s 8ms/step - loss: 0.0764 - acc: 0.9706 - val_loss: 0.3956 - val_acc: 0.8983
"""
