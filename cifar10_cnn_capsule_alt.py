"""Train a simple CNN-Capsule Network on the CIFAR10 small images dataset.

Without Data Augmentation:
It gets to 75% validation accuracy in 10 epochs,
and 79% after 15 epochs, and overfitting after 20 epochs

With Data Augmentation:
It gets to 75% validation accuracy in 10 epochs,
and 79% after 15 epochs, and 83% after 30 epcohs.
In my test, highest validation accuracy is 83.79% after 50 epcohs.

This is a fast Implement, just 20s/epcoh with a gtx 1070 gpu.
"""

from __future__ import print_function
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from Capsule import *
from SpatialPyramidPooling import SpatialPyramidPooling  #add
import os

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

# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
                
                
batch_size = 128
num_classes = 10
epochs = 1
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

def model_cifar(input_image=Input(shape=(None, None, 3))):
# A common Conv2D model
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(input_image)
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    x = BatchNormalization(axis=3)(x)  
    x = Dropout(0.5)(x)                
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = BatchNormalization(axis=3)(x)  
    x = Dropout(0.5)(x)                
    x = AveragePooling2D((2, 2))(x)    
    x = Conv2D(256, (3, 3), activation='relu',padding='same')(x)  
    x = Conv2D(256, (3, 3), activation='relu',padding='same')(x)  
    #x = BatchNormalization(axis=3)(x)  
    x = Dropout(0.5)(x)                
    return x,input_image

"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
then connect a Capsule layer.

the output of final model is the lengths of 10 Capsule, whose dim=16.

the length of Capsule is the proba,
so the problem becomes a 10 two-classification problem.
"""

#x,input_image=model_cifar(input_image=Input(shape=(32, 32, 3)))

#SPP
x1,input_image1=model_cifar(input_image=Input(shape=(None, None, 3)))
x1 = SpatialPyramidPooling([1,2,4])(x1)    #[1,2,4]
output1 = Dense(num_classes, activation='softmax')(x1)

#AveragePooling
x2,input_image2=model_cifar(input_image=Input(shape=(32, 32, 3)))
x2 = AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='tf')(x2)
x2 = Flatten()(x2)
output2 = Dense(num_classes, activation='softmax')(x2)

#Capsule
x3,input_image3=model_cifar(input_image=Input(shape=(None, None, 3)))
x3 = Reshape((-1, 128))(x3)
capsule = Capsule(10, 96, 3, True)(x3)  #16
output3 = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

model1 = Model(inputs=input_image1, outputs=output1)
model2 = Model(inputs=input_image2, outputs=output2)
model3 = Model(inputs=input_image3, outputs=output3)

# we use a margin loss
model1.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
model2.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
model3.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.summary()
model2.summary()
model3.summary()

# we can compare the performance with or without data augmentation
data_augmentation = True

#model.load_weights('params_CapsAve_epoch_000.hdf5')

if not data_augmentation:
    print('Not using data augmentation.')
    for j in range(100):
        print("*****j= ",j)
        history=model1.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)
        model1.save_weights('params_CapsAve1_epoch_{0:03d}.hdf5'.format(0), True)
        save_history(history, os.path.join("./", 'history_CapsAve1.txt'),0)
        
        history=model2.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)
        model2.save_weights('params_CapsAve2_epoch_{0:03d}.hdf5'.format(0), True)
        save_history(history, os.path.join("./", 'history_CapsAve2.txt'),0)
               
        history=model3.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)
        model3.save_weights('params_CapsAve3_epoch_{0:03d}.hdf5'.format(0), True)
        save_history(history, os.path.join("./", 'history_CapsAve3.txt'),0)    
            
else:
    print('Using real-time data augmentation.')
    for j in range(100):
        print("*****j= ",j)
    # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by dataset std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in 0 to 180 degrees
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
    
    # Fit the model on the batches generated by datagen.flow().
        history=model1.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_test, y_test),
            workers=4)
        model1.save_weights('params_CapsAve1_epoch_{0:03d}.hdf5'.format(0), True)
        save_history(history, os.path.join("./", 'history_CapsAve1.txt'),0)
        
        history=model2.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_test, y_test),
            workers=4)
        model2.save_weights('params_CapsAve2_epoch_{0:03d}.hdf5'.format(0), True)
        save_history(history, os.path.join("./", 'history_CapsAve2.txt'),0)
        
        history=model3.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_test, y_test),
            workers=4)
        model3.save_weights('params_CapsAve3_epoch_{0:03d}.hdf5'.format(0), True)
        save_history(history, os.path.join("./", 'history_CapsAve3.txt'),0)    

"""
x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid', dim_ordering='tf')(x)
x = Flatten()(x)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 30, 64)        1792
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 64)        36928
_________________________________________________________________
average_pooling2d_1 (Average (None, 14, 14, 64)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 128)       73856
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 10, 128)       147584
_________________________________________________________________
average_pooling2d_2 (Average (None, 1, 1, 128)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 261,450
Trainable params: 261,450
Non-trainable params: 0
_________________________________________________________________
Using real-time data augmentation.
Epoch 1/100
2018-04-08 13:27:01.050320: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2018-04-08 13:27:01.320096: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1212] Found device 0 with properties:
name: GeForce GTX 1060 3GB major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 3.00GiB freeMemory: 2.44GiB
2018-04-08 13:27:01.323388: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1312] Adding visible gpu devices: 0
2018-04-08 13:27:01.834654: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2148 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 3GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
391/391 [==============================] - 15s 39ms/step - loss: 1.8253 - acc: 0.3092 - val_loss: 1.6237 - val_acc: 0.3937
Epoch 2/100
391/391 [==============================] - 13s 33ms/step - loss: 1.5676 - acc: 0.4223 - val_loss: 1.4000 - val_acc: 0.4882
Epoch 3/100
391/391 [==============================] - 13s 33ms/step - loss: 1.4273 - acc: 0.4855 - val_loss: 1.3678 - val_acc: 0.5162
Epoch 10/100
391/391 [==============================] - 13s 33ms/step - loss: 1.0025 - acc: 0.6492 - val_loss: 0.9563 - val_acc: 0.6616
Epoch 50/100
391/391 [==============================] - 13s 33ms/step - loss: 0.4948 - acc: 0.8278 - val_loss: 0.6609 - val_acc: 0.7898
Epoch 100/100
391/391 [==============================] - 13s 33ms/step - loss: 0.3103 - acc: 0.8916 - val_loss: 0.6338 - val_acc: 0.8190

SPPAve[1]
spatial_pyramid_pooling_1 (S (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 261,450
Trainable params: 261,450
Non-trainable params: 0
_________________________________________________________________
Using real-time data augmentation.
Epoch 1/100
2018-04-08 13:00:14.304692: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2018-04-08 13:00:14.566223: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1212] Found device 0 with properties:
name: GeForce GTX 1060 3GB major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 3.00GiB freeMemory: 2.44GiB
2018-04-08 13:00:14.569823: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1312] Adding visible gpu devices: 0
2018-04-08 13:00:15.060817: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2148 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 3GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
391/391 [==============================] - 15s 39ms/step - loss: 1.8385 - acc: 0.3047 - val_loss: 1.6173 - val_acc: 0.3990
Epoch 2/100
391/391 [==============================] - 13s 34ms/step - loss: 1.5910 - acc: 0.4143 - val_loss: 1.4401 - val_acc: 0.4734
Epoch 3/100
391/391 [==============================] - 13s 33ms/step - loss: 1.4643 - acc: 0.4695 - val_loss: 1.3739 - val_acc: 0.5137
Epoch 4/100
391/391 [==============================] - 13s 33ms/step - loss: 1.3601 - acc: 0.5134 - val_loss: 1.2849 - val_acc: 0.5321
Epoch 5/100
391/391 [==============================] - 13s 34ms/step - loss: 1.2846 - acc: 0.5413 - val_loss: 1.2091 - val_acc: 0.5686
Epoch 6/100
391/391 [==============================] - 13s 34ms/step - loss: 1.2248 - acc: 0.5647 - val_loss: 1.1741 - val_acc: 0.5838
Epoch 7/100
391/391 [==============================] - 13s 34ms/step - loss: 1.1619 - acc: 0.5886 - val_loss: 1.1715 - val_acc: 0.5906
Epoch 8/100
391/391 [==============================] - 13s 33ms/step - loss: 1.1167 - acc: 0.6043 - val_loss: 1.0819 - val_acc: 0.6161
Epoch 9/100
391/391 [==============================] - 13s 34ms/step - loss: 1.0734 - acc: 0.6219 - val_loss: 1.0499 - val_acc: 0.6306
Epoch 10/100
391/391 [==============================] - 13s 34ms/step - loss: 1.0353 - acc: 0.6341 - val_loss: 1.1088 - val_acc: 0.6223
Epoch 11/100
391/391 [==============================] - 13s 34ms/step - loss: 1.0000 - acc: 0.6476 - val_loss: 1.0049 - val_acc: 0.6510
Epoch 12/100
391/391 [==============================] - 13s 34ms/step - loss: 0.9654 - acc: 0.6601 - val_loss: 0.9355 - val_acc: 0.6735
Epoch 13/100
391/391 [==============================] - 14s 35ms/step - loss: 0.9357 - acc: 0.6735 - val_loss: 0.9893 - val_acc: 0.6578
Epoch 14/100
391/391 [==============================] - 13s 34ms/step - loss: 0.9115 - acc: 0.6792 - val_loss: 0.9404 - val_acc: 0.6741
Epoch 15/100
391/391 [==============================] - 13s 34ms/step - loss: 0.8913 - acc: 0.6885 - val_loss: 0.8597 - val_acc: 0.7022
Epoch 16/100
391/391 [==============================] - 13s 34ms/step - loss: 0.8642 - acc: 0.6987 - val_loss: 0.8854 - val_acc: 0.6995
Epoch 17/100
391/391 [==============================] - 13s 34ms/step - loss: 0.8390 - acc: 0.7080 - val_loss: 0.8774 - val_acc: 0.7001
Epoch 18/100
391/391 [==============================] - 13s 34ms/step - loss: 0.8240 - acc: 0.7125 - val_loss: 0.8059 - val_acc: 0.7225
Epoch 19/100
391/391 [==============================] - 13s 34ms/step - loss: 0.8006 - acc: 0.7198 - val_loss: 0.8366 - val_acc: 0.7154
Epoch 20/100
391/391 [==============================] - 13s 34ms/step - loss: 0.7803 - acc: 0.7293 - val_loss: 0.7962 - val_acc: 0.7266
Epoch 21/100
391/391 [==============================] - 13s 34ms/step - loss: 0.7680 - acc: 0.7333 - val_loss: 0.8197 - val_acc: 0.7221
Epoch 22/100
391/391 [==============================] - 13s 34ms/step - loss: 0.7442 - acc: 0.7406 - val_loss: 0.7635 - val_acc: 0.7362
Epoch 23/100
391/391 [==============================] - 13s 34ms/step - loss: 0.7320 - acc: 0.7445 - val_loss: 0.8250 - val_acc: 0.7289
Epoch 24/100
391/391 [==============================] - 13s 34ms/step - loss: 0.7195 - acc: 0.7499 - val_loss: 0.7400 - val_acc: 0.7452
Epoch 25/100
391/391 [==============================] - 13s 34ms/step - loss: 0.7048 - acc: 0.7539 - val_loss: 0.7355 - val_acc: 0.7527
Epoch 26/100
391/391 [==============================] - 13s 34ms/step - loss: 0.6900 - acc: 0.7608 - val_loss: 0.7228 - val_acc: 0.7555
Epoch 27/100
391/391 [==============================] - 13s 34ms/step - loss: 0.6813 - acc: 0.7638 - val_loss: 0.7118 - val_acc: 0.7531
Epoch 28/100
391/391 [==============================] - 13s 34ms/step - loss: 0.6647 - acc: 0.7692 - val_loss: 0.7316 - val_acc: 0.7563
Epoch 29/100
391/391 [==============================] - 13s 34ms/step - loss: 0.6588 - acc: 0.7705 - val_loss: 0.7522 - val_acc: 0.7509
Epoch 30/100
391/391 [==============================] - 13s 34ms/step - loss: 0.6397 - acc: 0.7769 - val_loss: 0.7429 - val_acc: 0.7528
Epoch 31/100
391/391 [==============================] - 13s 34ms/step - loss: 0.6303 - acc: 0.7823 - val_loss: 0.6794 - val_acc: 0.7661
Epoch 32/100
391/391 [==============================] - 13s 34ms/step - loss: 0.6205 - acc: 0.7832 - val_loss: 0.8105 - val_acc: 0.7384
Epoch 33/100
391/391 [==============================] - 13s 34ms/step - loss: 0.6151 - acc: 0.7874 - val_loss: 0.6538 - val_acc: 0.7838
Epoch 34/100
391/391 [==============================] - 13s 34ms/step - loss: 0.6016 - acc: 0.7932 - val_loss: 0.6975 - val_acc: 0.7700
Epoch 35/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5911 - acc: 0.7936 - val_loss: 0.6920 - val_acc: 0.7704
Epoch 36/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5820 - acc: 0.7990 - val_loss: 0.6515 - val_acc: 0.7832
Epoch 37/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5732 - acc: 0.8005 - val_loss: 0.6388 - val_acc: 0.7859
Epoch 38/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5704 - acc: 0.8027 - val_loss: 0.7028 - val_acc: 0.7635
Epoch 39/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5627 - acc: 0.8038 - val_loss: 0.6710 - val_acc: 0.7792
Epoch 40/100
391/391 [==============================] - 13s 33ms/step - loss: 0.5490 - acc: 0.8092 - val_loss: 0.6558 - val_acc: 0.7819
Epoch 41/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5443 - acc: 0.8123 - val_loss: 0.6984 - val_acc: 0.7739
Epoch 42/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5366 - acc: 0.8142 - val_loss: 0.6497 - val_acc: 0.7864
Epoch 43/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5305 - acc: 0.8156 - val_loss: 0.6449 - val_acc: 0.7909
Epoch 44/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5218 - acc: 0.8207 - val_loss: 0.6361 - val_acc: 0.7947
Epoch 45/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5219 - acc: 0.8193 - val_loss: 0.6473 - val_acc: 0.7882
Epoch 46/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5068 - acc: 0.8240 - val_loss: 0.6108 - val_acc: 0.7980
Epoch 47/100
391/391 [==============================] - 13s 34ms/step - loss: 0.5041 - acc: 0.8263 - val_loss: 0.6381 - val_acc: 0.7913
Epoch 48/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4889 - acc: 0.8311 - val_loss: 0.6012 - val_acc: 0.7996
Epoch 49/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4906 - acc: 0.8293 - val_loss: 0.6153 - val_acc: 0.7984
Epoch 50/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4880 - acc: 0.8320 - val_loss: 0.5906 - val_acc: 0.8062
Epoch 51/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4758 - acc: 0.8341 - val_loss: 0.5995 - val_acc: 0.8086
Epoch 52/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4747 - acc: 0.8355 - val_loss: 0.6388 - val_acc: 0.7912
Epoch 53/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4718 - acc: 0.8376 - val_loss: 0.6346 - val_acc: 0.7941
Epoch 54/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4593 - acc: 0.8420 - val_loss: 0.6625 - val_acc: 0.7889
Epoch 55/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4591 - acc: 0.8398 - val_loss: 0.5805 - val_acc: 0.8063
Epoch 56/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4545 - acc: 0.8427 - val_loss: 0.6032 - val_acc: 0.8064
Epoch 57/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4453 - acc: 0.8452 - val_loss: 0.6523 - val_acc: 0.7945
Epoch 58/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4468 - acc: 0.8453 - val_loss: 0.5950 - val_acc: 0.8032
Epoch 59/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4353 - acc: 0.8510 - val_loss: 0.5827 - val_acc: 0.8045
Epoch 60/100
391/391 [==============================] - 13s 33ms/step - loss: 0.4368 - acc: 0.8480 - val_loss: 0.5976 - val_acc: 0.8065
Epoch 61/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4284 - acc: 0.8499 - val_loss: 0.6275 - val_acc: 0.8026
Epoch 62/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4183 - acc: 0.8529 - val_loss: 0.5830 - val_acc: 0.8106
Epoch 63/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4172 - acc: 0.8559 - val_loss: 0.5777 - val_acc: 0.8153
Epoch 64/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4117 - acc: 0.8573 - val_loss: 0.6543 - val_acc: 0.7959
Epoch 65/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4098 - acc: 0.8570 - val_loss: 0.6068 - val_acc: 0.8122
Epoch 66/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4030 - acc: 0.8588 - val_loss: 0.5953 - val_acc: 0.8133
Epoch 67/100
391/391 [==============================] - 13s 34ms/step - loss: 0.4050 - acc: 0.8580 - val_loss: 0.5771 - val_acc: 0.8171
Epoch 68/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3979 - acc: 0.8627 - val_loss: 0.6313 - val_acc: 0.7984
Epoch 69/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3931 - acc: 0.8618 - val_loss: 0.6151 - val_acc: 0.8057
Epoch 70/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3887 - acc: 0.8660 - val_loss: 0.5945 - val_acc: 0.8107
Epoch 71/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3842 - acc: 0.8647 - val_loss: 0.6134 - val_acc: 0.8130
Epoch 72/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3787 - acc: 0.8667 - val_loss: 0.5810 - val_acc: 0.8160
Epoch 73/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3761 - acc: 0.8705 - val_loss: 0.5934 - val_acc: 0.8157
Epoch 74/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3659 - acc: 0.8729 - val_loss: 0.5973 - val_acc: 0.8153
Epoch 75/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3661 - acc: 0.8727 - val_loss: 0.5710 - val_acc: 0.8212
Epoch 76/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3652 - acc: 0.8715 - val_loss: 0.5943 - val_acc: 0.8176
Epoch 77/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3611 - acc: 0.8742 - val_loss: 0.6157 - val_acc: 0.8113
Epoch 78/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3602 - acc: 0.8739 - val_loss: 0.5597 - val_acc: 0.8233
Epoch 79/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3552 - acc: 0.8761 - val_loss: 0.5654 - val_acc: 0.8249
Epoch 80/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3480 - acc: 0.8772 - val_loss: 0.6060 - val_acc: 0.8178
Epoch 81/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3467 - acc: 0.8797 - val_loss: 0.5585 - val_acc: 0.8259
Epoch 82/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3431 - acc: 0.8802 - val_loss: 0.5803 - val_acc: 0.8209
Epoch 83/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3401 - acc: 0.8815 - val_loss: 0.6023 - val_acc: 0.8156
Epoch 84/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3342 - acc: 0.8824 - val_loss: 0.6468 - val_acc: 0.8140
Epoch 85/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3308 - acc: 0.8837 - val_loss: 0.5920 - val_acc: 0.8197
Epoch 86/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3312 - acc: 0.8857 - val_loss: 0.5933 - val_acc: 0.8159
Epoch 87/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3257 - acc: 0.8865 - val_loss: 0.6061 - val_acc: 0.8193
Epoch 88/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3213 - acc: 0.8877 - val_loss: 0.5876 - val_acc: 0.8219
Epoch 89/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3199 - acc: 0.8877 - val_loss: 0.6672 - val_acc: 0.8067
Epoch 90/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3135 - acc: 0.8900 - val_loss: 0.5909 - val_acc: 0.8234
Epoch 91/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3129 - acc: 0.8901 - val_loss: 0.6488 - val_acc: 0.8096
Epoch 92/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3144 - acc: 0.8890 - val_loss: 0.6542 - val_acc: 0.8122
Epoch 93/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3113 - acc: 0.8901 - val_loss: 0.6108 - val_acc: 0.8177
Epoch 94/100
391/391 [==============================] - 13s 33ms/step - loss: 0.3047 - acc: 0.8935 - val_loss: 0.6064 - val_acc: 0.8211
Epoch 95/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3099 - acc: 0.8922 - val_loss: 0.6591 - val_acc: 0.8109
Epoch 96/100
391/391 [==============================] - 13s 34ms/step - loss: 0.2986 - acc: 0.8955 - val_loss: 0.6429 - val_acc: 0.8164
Epoch 97/100
391/391 [==============================] - 13s 34ms/step - loss: 0.3043 - acc: 0.8924 - val_loss: 0.6249 - val_acc: 0.8140
Epoch 98/100
391/391 [==============================] - 13s 34ms/step - loss: 0.2959 - acc: 0.8968 - val_loss: 0.6395 - val_acc: 0.8196
Epoch 99/100
391/391 [==============================] - 13s 34ms/step - loss: 0.2957 - acc: 0.8964 - val_loss: 0.6167 - val_acc: 0.8197
Epoch 100/100
391/391 [==============================] - 13s 34ms/step - loss: 0.2916 - acc: 0.8970 - val_loss: 0.6049 - val_acc: 0.8212

Capsule_original
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, None, None, 3)     0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, None, None, 64)    1792
_________________________________________________________________
conv2d_2 (Conv2D)            (None, None, None, 64)    36928
_________________________________________________________________
average_pooling2d_1 (Average (None, None, None, 64)    0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, None, None, 128)   73856
_________________________________________________________________
conv2d_4 (Conv2D)            (None, None, None, 128)   147584
_________________________________________________________________
reshape_1 (Reshape)          (None, None, 128)         0
_________________________________________________________________
capsule_1 (Capsule)          (None, 10, 16)            20480
_________________________________________________________________
lambda_1 (Lambda)            (None, 10)                0
=================================================================
Total params: 280,640
Trainable params: 280,640
Non-trainable params: 0
_________________________________________________________________
Using real-time data augmentation.
Epoch 1/100
2018-04-08 11:59:47.375498: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2018-04-08 11:59:47.641859: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1212] Found device 0 with properties:
name: GeForce GTX 1060 3GB major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 3.00GiB freeMemory: 2.44GiB
2018-04-08 11:59:47.646450: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1312] Adding visible gpu devices: 0
2018-04-08 11:59:48.155209: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2148 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 3GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
391/391 [==============================] - 19s 48ms/step - loss: 0.4257 - acc: 0.3484 - val_loss: 0.3586 - val_acc: 0.4782
Epoch 2/100
391/391 [==============================] - 16s 42ms/step - loss: 0.3389 - acc: 0.5114 - val_loss: 0.3133 - val_acc: 0.5600
Epoch 3/100
391/391 [==============================] - 16s 42ms/step - loss: 0.2976 - acc: 0.5850 - val_loss: 0.2721 - val_acc: 0.6253
Epoch 4/100
391/391 [==============================] - 16s 42ms/step - loss: 0.2697 - acc: 0.6301 - val_loss: 0.2472 - val_acc: 0.6610
Epoch 5/100
391/391 [==============================] - 16s 42ms/step - loss: 0.2481 - acc: 0.6652 - val_loss: 0.2342 - val_acc: 0.6828
Epoch 6/100
391/391 [==============================] - 16s 42ms/step - loss: 0.2318 - acc: 0.6900 - val_loss: 0.2214 - val_acc: 0.7014
Epoch 7/100
391/391 [==============================] - 17s 42ms/step - loss: 0.2205 - acc: 0.7093 - val_loss: 0.2168 - val_acc: 0.7166
Epoch 8/100
391/391 [==============================] - 16s 41ms/step - loss: 0.2098 - acc: 0.7254 - val_loss: 0.2133 - val_acc: 0.7166
Epoch 9/100
391/391 [==============================] - 16s 42ms/step - loss: 0.2023 - acc: 0.7391 - val_loss: 0.1929 - val_acc: 0.7495
Epoch 10/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1953 - acc: 0.7489 - val_loss: 0.2028 - val_acc: 0.7323
Epoch 11/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1867 - acc: 0.7624 - val_loss: 0.1943 - val_acc: 0.7533
Epoch 12/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1813 - acc: 0.7695 - val_loss: 0.1951 - val_acc: 0.7496
Epoch 13/100
391/391 [==============================] - 17s 42ms/step - loss: 0.1767 - acc: 0.7784 - val_loss: 0.1849 - val_acc: 0.7592
Epoch 14/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1720 - acc: 0.7856 - val_loss: 0.1780 - val_acc: 0.7715
Epoch 15/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1672 - acc: 0.7927 - val_loss: 0.1721 - val_acc: 0.7828
Epoch 16/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1621 - acc: 0.8035 - val_loss: 0.1722 - val_acc: 0.7867
Epoch 17/100
391/391 [==============================] - 17s 42ms/step - loss: 0.1596 - acc: 0.8062 - val_loss: 0.1678 - val_acc: 0.7920
Epoch 18/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1556 - acc: 0.8129 - val_loss: 0.1743 - val_acc: 0.7751
Epoch 19/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1525 - acc: 0.8166 - val_loss: 0.1638 - val_acc: 0.7954
Epoch 20/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1496 - acc: 0.8212 - val_loss: 0.1719 - val_acc: 0.7915
Epoch 21/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1479 - acc: 0.8226 - val_loss: 0.1612 - val_acc: 0.8011
Epoch 22/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1450 - acc: 0.8288 - val_loss: 0.1646 - val_acc: 0.7968
Epoch 23/100
391/391 [==============================] - 17s 42ms/step - loss: 0.1425 - acc: 0.8335 - val_loss: 0.1612 - val_acc: 0.8029
Epoch 24/100
391/391 [==============================] - 17s 42ms/step - loss: 0.1394 - acc: 0.8369 - val_loss: 0.1591 - val_acc: 0.8034
Epoch 25/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1387 - acc: 0.8372 - val_loss: 0.1638 - val_acc: 0.7940
Epoch 26/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1362 - acc: 0.8416 - val_loss: 0.1623 - val_acc: 0.8034
Epoch 27/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1339 - acc: 0.8465 - val_loss: 0.1675 - val_acc: 0.7985
Epoch 28/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1331 - acc: 0.8453 - val_loss: 0.1622 - val_acc: 0.8033
Epoch 29/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1302 - acc: 0.8504 - val_loss: 0.1581 - val_acc: 0.8098
Epoch 30/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1292 - acc: 0.8535 - val_loss: 0.1536 - val_acc: 0.8146
Epoch 31/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1286 - acc: 0.8525 - val_loss: 0.1604 - val_acc: 0.8092
Epoch 32/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1249 - acc: 0.8609 - val_loss: 0.1510 - val_acc: 0.8228
Epoch 33/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1238 - acc: 0.8616 - val_loss: 0.1592 - val_acc: 0.8124
Epoch 34/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1235 - acc: 0.8611 - val_loss: 0.1590 - val_acc: 0.8092
Epoch 35/100
391/391 [==============================] - 17s 42ms/step - loss: 0.1210 - acc: 0.8641 - val_loss: 0.1630 - val_acc: 0.8042
Epoch 36/100
391/391 [==============================] - 17s 42ms/step - loss: 0.1192 - acc: 0.8688 - val_loss: 0.1594 - val_acc: 0.8047
Epoch 37/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1188 - acc: 0.8685 - val_loss: 0.1557 - val_acc: 0.8095
Epoch 38/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1168 - acc: 0.8724 - val_loss: 0.1518 - val_acc: 0.8195
Epoch 39/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1157 - acc: 0.8745 - val_loss: 0.1539 - val_acc: 0.8121
Epoch 40/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1152 - acc: 0.8745 - val_loss: 0.1532 - val_acc: 0.8163
Epoch 41/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1129 - acc: 0.8781 - val_loss: 0.1502 - val_acc: 0.8147
Epoch 42/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1113 - acc: 0.8810 - val_loss: 0.1537 - val_acc: 0.8161
Epoch 43/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1111 - acc: 0.8811 - val_loss: 0.1551 - val_acc: 0.8087
Epoch 44/100
391/391 [==============================] - 17s 42ms/step - loss: 0.1101 - acc: 0.8835 - val_loss: 0.1471 - val_acc: 0.8229
Epoch 45/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1091 - acc: 0.8843 - val_loss: 0.1487 - val_acc: 0.8172
Epoch 46/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1079 - acc: 0.8859 - val_loss: 0.1508 - val_acc: 0.8166
Epoch 47/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1058 - acc: 0.8890 - val_loss: 0.1482 - val_acc: 0.8221
Epoch 48/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1046 - acc: 0.8919 - val_loss: 0.1514 - val_acc: 0.8148
Epoch 49/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1053 - acc: 0.8895 - val_loss: 0.1526 - val_acc: 0.8187
Epoch 50/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1045 - acc: 0.8906 - val_loss: 0.1560 - val_acc: 0.8137
Epoch 51/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1032 - acc: 0.8929 - val_loss: 0.1496 - val_acc: 0.8188
Epoch 52/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1023 - acc: 0.8938 - val_loss: 0.1474 - val_acc: 0.8229
Epoch 53/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1010 - acc: 0.8963 - val_loss: 0.1529 - val_acc: 0.8159
Epoch 54/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0997 - acc: 0.8998 - val_loss: 0.1496 - val_acc: 0.8165
Epoch 55/100
391/391 [==============================] - 16s 42ms/step - loss: 0.1003 - acc: 0.8970 - val_loss: 0.1496 - val_acc: 0.8197
Epoch 56/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0998 - acc: 0.8980 - val_loss: 0.1500 - val_acc: 0.8234
Epoch 57/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0984 - acc: 0.9007 - val_loss: 0.1543 - val_acc: 0.8099
Epoch 58/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0980 - acc: 0.9002 - val_loss: 0.1553 - val_acc: 0.8116
Epoch 59/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0973 - acc: 0.9038 - val_loss: 0.1477 - val_acc: 0.8249
Epoch 60/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0962 - acc: 0.9048 - val_loss: 0.1530 - val_acc: 0.8170
Epoch 61/100
391/391 [==============================] - 17s 43ms/step - loss: 0.0952 - acc: 0.9046 - val_loss: 0.1515 - val_acc: 0.8210
Epoch 62/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0949 - acc: 0.9043 - val_loss: 0.1614 - val_acc: 0.8085
Epoch 63/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0943 - acc: 0.9071 - val_loss: 0.1457 - val_acc: 0.8268
Epoch 64/100
391/391 [==============================] - 17s 43ms/step - loss: 0.0936 - acc: 0.9083 - val_loss: 0.1514 - val_acc: 0.8200
Epoch 65/100
391/391 [==============================] - 17s 43ms/step - loss: 0.0938 - acc: 0.9064 - val_loss: 0.1541 - val_acc: 0.8197
Epoch 66/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0929 - acc: 0.9091 - val_loss: 0.1445 - val_acc: 0.8271
Epoch 67/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0914 - acc: 0.9113 - val_loss: 0.1533 - val_acc: 0.8186
Epoch 68/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0922 - acc: 0.9109 - val_loss: 0.1465 - val_acc: 0.8231
Epoch 69/100
391/391 [==============================] - 17s 43ms/step - loss: 0.0902 - acc: 0.9135 - val_loss: 0.1537 - val_acc: 0.8175
Epoch 70/100
391/391 [==============================] - 17s 43ms/step - loss: 0.0903 - acc: 0.9118 - val_loss: 0.1499 - val_acc: 0.8245
Epoch 71/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0915 - acc: 0.9118 - val_loss: 0.1496 - val_acc: 0.8212
Epoch 72/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0887 - acc: 0.9151 - val_loss: 0.1522 - val_acc: 0.8200
Epoch 73/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0881 - acc: 0.9161 - val_loss: 0.1492 - val_acc: 0.8232
Epoch 74/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0879 - acc: 0.9159 - val_loss: 0.1508 - val_acc: 0.8218
Epoch 75/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0871 - acc: 0.9184 - val_loss: 0.1562 - val_acc: 0.8166
Epoch 76/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0875 - acc: 0.9181 - val_loss: 0.1552 - val_acc: 0.8188
Epoch 77/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0868 - acc: 0.9178 - val_loss: 0.1527 - val_acc: 0.8119
Epoch 78/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0870 - acc: 0.9164 - val_loss: 0.1549 - val_acc: 0.8152
Epoch 79/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0857 - acc: 0.9202 - val_loss: 0.1526 - val_acc: 0.8184
Epoch 80/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0849 - acc: 0.9214 - val_loss: 0.1582 - val_acc: 0.8099
Epoch 81/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0857 - acc: 0.9201 - val_loss: 0.1531 - val_acc: 0.8193
Epoch 82/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0837 - acc: 0.9232 - val_loss: 0.1570 - val_acc: 0.8084
Epoch 83/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0836 - acc: 0.9225 - val_loss: 0.1503 - val_acc: 0.8202
Epoch 84/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0836 - acc: 0.9215 - val_loss: 0.1513 - val_acc: 0.8232
Epoch 85/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0833 - acc: 0.9237 - val_loss: 0.1523 - val_acc: 0.8106
Epoch 86/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0826 - acc: 0.9239 - val_loss: 0.1594 - val_acc: 0.8070
Epoch 87/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0830 - acc: 0.9241 - val_loss: 0.1499 - val_acc: 0.8235
Epoch 88/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0815 - acc: 0.9253 - val_loss: 0.1472 - val_acc: 0.8291
Epoch 89/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0813 - acc: 0.9265 - val_loss: 0.1538 - val_acc: 0.8235
Epoch 90/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0816 - acc: 0.9256 - val_loss: 0.1517 - val_acc: 0.8202
Epoch 91/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0813 - acc: 0.9257 - val_loss: 0.1545 - val_acc: 0.8151
Epoch 92/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0800 - acc: 0.9292 - val_loss: 0.1520 - val_acc: 0.8200
Epoch 93/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0800 - acc: 0.9274 - val_loss: 0.1549 - val_acc: 0.8211
Epoch 94/100
391/391 [==============================] - 17s 43ms/step - loss: 0.0799 - acc: 0.9283 - val_loss: 0.1531 - val_acc: 0.8151
Epoch 95/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0798 - acc: 0.9275 - val_loss: 0.1556 - val_acc: 0.8153
Epoch 96/100
391/391 [==============================] - 16s 42ms/step - loss: 0.0785 - acc: 0.9314 - val_loss: 0.1523 - val_acc: 0.8235
Epoch 97/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0789 - acc: 0.9291 - val_loss: 0.1552 - val_acc: 0.8236
Epoch 98/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0788 - acc: 0.9294 - val_loss: 0.1470 - val_acc: 0.8266
Epoch 99/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0774 - acc: 0.9315 - val_loss: 0.1479 - val_acc: 0.8272
Epoch 100/100
391/391 [==============================] - 17s 42ms/step - loss: 0.0776 - acc: 0.9320 - val_loss: 0.1470 - val_acc: 0.8309
"""