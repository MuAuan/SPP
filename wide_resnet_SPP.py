import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, AveragePooling2D, BatchNormalization, Dropout
from keras.engine import Input, Model
from keras.layers import merge
from keras.optimizers import SGD,Adam
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K
import json
import time
from SpatialPyramidPooling import SpatialPyramidPooling  #add

nb_classes = 10

def save_history(history, result_file,epochs):
    loss = history.history['loss']
    #conv_loss = history.history['conv_out_loss']
    #acc = history.history['conv_out_acc']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    #val_conv_loss = history.history['val_out_recon_loss']
    #val_acc = history.history['val_conv_out_acc']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "a") as fp:
        if epochs==0:
            fp.write("i\tloss\tacc\tval_loss\tval_acc\n")
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\n" % (epochs, loss[i], acc[i], val_loss[i], val_acc[i]))
        else:
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\n" % (epochs, loss[i], acc[i], val_loss[i], val_acc[i]))


def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def residual_block(x, nb_filters=16, subsample_factor=1):
    
    prev_nb_channels = K.int_shape(x)[3]

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample, dim_ordering='tf')(x)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x
        
    if nb_filters > prev_nb_channels:
        shortcut = Lambda(zero_pad_channels,
                          arguments={'pad': nb_filters - prev_nb_channels})(shortcut)

    y = BatchNormalization(axis=3)(x)
    y = Activation('relu')(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=subsample,
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)
    
    out = merge([y, shortcut], mode='sum')

    return out

#%%time

img_rows, img_cols = None, None
img_channels = 3

blocks_per_group = 4
widening_factor = 5  #10

inputs = Input(shape=(img_rows, img_cols, img_channels))

x = Convolution2D(16, 3, 3, 
                  init='he_normal', border_mode='same', dim_ordering='tf')(inputs)

for i in range(0, blocks_per_group):
    nb_filters = 8 * widening_factor  #16
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)

for i in range(0, blocks_per_group):
    nb_filters = 16 * widening_factor  #32
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

for i in range(0, blocks_per_group):
    nb_filters = 32 * widening_factor  #64
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
#x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid', dim_ordering='tf')(x)
#x = Flatten()(x)
x = SpatialPyramidPooling([1, 2, 4])(x)

softmax = Dense(nb_classes, activation='softmax')(x)

model = Model(input=inputs, output=softmax)

model.summary()

#%%time

#sgd = SGD(lr=0.1, decay=5e-4, momentum=0.9, nesterov=True)
adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)

model.compile(optimizer=adm,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# reorder dimensions for tensorflow
x_train = np.transpose(x_train.astype('float32'), (0, 1, 2,3))
mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train)
x_train = x_train/255.  # (x_train - mean) / std
x_test = np.transpose(x_test.astype('float32'), (0, 1,2, 3))
x_test = x_test/255.  # (x_test - mean) / std
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

batch_size = 128
nb_epoch = 10
data_augmentation = True

# Learning rate schedule
def lr_sch(epoch):
    if epoch < 2:
        return 0.001
    elif epoch < 6:
        return 0.0001
    elif epoch < 8:
        return 0.00004
    else:
        return 0.000008

# Learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_sch)

def train(adm, model, data, epoch_size=32,batch_size=128):
    (x_train, y_train), (x_test, y_test) = data
    # Learning rate schedule
    #loss='binary_crossentropy'  'categorical_crossentropy' 'mse'
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, epochs=epoch_size, verbose=1,
                        validation_data=(x_test, y_test), shuffle=True,
                        callbacks=[lr_scheduler])

    return model,history

# Model saving callback
#checkpointer = ModelCheckpoint(filepath='stochastic_depth_cifar10.hdf5', verbose=1, save_best_only=True)

for j in range(20):
        model, history = train(adm,model=model, data=((x_train, y_train), (x_test, y_test)), epoch_size=nb_epoch ,batch_size=batch_size)
        model.save_weights('params_SPP_epoch_{0:03d}.hdf5'.format(j), True)
        plot_generated_batch(j,model=model, data1=(x_train, y_train),data2=(x_test, y_test))
        # 学習履歴を保存
        save_history(history, os.path.join("./", 'history_SPP_AE.txt'),j)

"""
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                        validation_data=(x_test, y_test), shuffle=True,
                        callbacks=[lr_scheduler])
else:
    print('Using real-time data augmentation.')

    # realtime data augmentation
    datagen_train = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False)
    datagen_train.fit(x_train)

    # fit the model on the batches generated by datagen.flow()
    history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                                  samples_per_epoch=x_train.shape[0], 
                                  nb_epoch=nb_epoch, verbose=1,
                                  validation_data=(x_test, y_test),
                                  callbacks=[lr_scheduler])

"""



