from __future__ import print_function
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
#from Capsule import *
#from SpatialPyramidPooling import SpatialPyramidPooling  #add
import os
import cv2

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
                
                
batch_size = 32
num_classes = 10
epochs = 1
img_rows, img_cols=96,96 #64,64
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
X_train =[]
X_test = []
for i in range(50000):
    dst = cv2.resize(x_train[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC) #cv2.INTER_LINEAR #cv2.INTER_CUBIC
    X_train.append(dst)
for i in range(10000):
    dst = cv2.resize(x_test[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
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

#x,input_image=model_cifar(input_image=Input(shape=(32, 32, 3)))

#AveragePooling
x2,input_image2=model_cifar(input_image=Input(shape=(img_rows, img_cols, 3)))
x2 = AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='tf')(x2)
x2 = Flatten()(x2)
output2 = Dense(num_classes, activation='softmax')(x2)

model2 = Model(inputs=input_image2, outputs=output2)

# we use a margin loss
model2.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()

# we can compare the performance with or without data augmentation
data_augmentation = True

#model.load_weights('params_CapsAve_epoch_000.hdf5')

if not data_augmentation:
    print('Not using data augmentation.')
    for j in range(100):
        print("*****j= ",j)
        history=model2.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)
        model2.save_weights('params_CapsAve2_epoch_{0:03d}.hdf5'.format(0), True)
        save_history(history, os.path.join("./", 'history_CapsAve2.txt'),0)
            
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
    
        history=model2.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_test, y_test),
            workers=4)
        model2.save_weights('params_CapsAve2_epoch_{0:03d}.hdf5'.format(0), True)
        save_history(history, os.path.join("./", 'history_CapsAve2.txt'),0)

        
"""
(32,32,3)
*****j=  88
Epoch 1/1
391/391 [==============================] - 23s 60ms/step - loss: 0.0673 - acc: 0.9186 - val_loss: 0.0982 - val_acc: 0.8898
*****j=  89
Epoch 1/1
391/391 [==============================] - 23s 60ms/step - loss: 0.0662 - acc: 0.9202 - val_loss: 0.1028 - val_acc: 0.8848
*****j=  90
Epoch 1/1
391/391 [==============================] - 23s 60ms/step - loss: 0.0650 - acc: 0.9222 - val_loss: 0.1106 - val_acc: 0.8744
*****j=  91
Epoch 1/1
391/391 [==============================] - 23s 59ms/step - loss: 0.0654 - acc: 0.9216 - val_loss: 0.1113 - val_acc: 0.8761
*****j=  92
Epoch 1/1
391/391 [==============================] - 23s 59ms/step - loss: 0.0636 - acc: 0.9227 - val_loss: 0.1063 - val_acc: 0.8799
*****j=  93
Epoch 1/1
391/391 [==============================] - 23s 59ms/step - loss: 0.0639 - acc: 0.9226 - val_loss: 0.1062 - val_acc: 0.8825
*****j=  94
Epoch 1/1
391/391 [==============================] - 23s 59ms/step - loss: 0.0641 - acc: 0.9228 - val_loss: 0.1040 - val_acc: 0.8839
*****j=  95
Epoch 1/1
391/391 [==============================] - 23s 59ms/step - loss: 0.0656 - acc: 0.9210 - val_loss: 0.1147 - val_acc: 0.8708
*****j=  96
Epoch 1/1
391/391 [==============================] - 23s 59ms/step - loss: 0.0639 - acc: 0.9225 - val_loss: 0.1069 - val_acc: 0.8809
*****j=  97
Epoch 1/1
391/391 [==============================] - 23s 59ms/step - loss: 0.0633 - acc: 0.9242 - val_loss: 0.1092 - val_acc: 0.8791
*****j=  98
Epoch 1/1
391/391 [==============================] - 23s 59ms/step - loss: 0.0603 - acc: 0.9281 - val_loss: 0.1132 - val_acc: 0.8766
*****j=  99
Epoch 1/1
391/391 [==============================] - 23s 59ms/step - loss: 0.0620 - acc: 0.9251 - val_loss: 0.1097 - val_acc: 0.8769

(64,64,3)
*****j=  88
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0609 - acc: 0.9261 - val_loss: 0.0977 - val_acc: 0.8918
*****j=  89
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0617 - acc: 0.9264 - val_loss: 0.1149 - val_acc: 0.8746
*****j=  90
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0639 - acc: 0.9228 - val_loss: 0.1048 - val_acc: 0.8840
*****j=  91
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0614 - acc: 0.9267 - val_loss: 0.0988 - val_acc: 0.8895
*****j=  92
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0590 - acc: 0.9293 - val_loss: 0.1011 - val_acc: 0.8872
*****j=  93
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0591 - acc: 0.9299 - val_loss: 0.1043 - val_acc: 0.8866
*****j=  94
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0595 - acc: 0.9284 - val_loss: 0.1031 - val_acc: 0.8858
*****j=  95
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0590 - acc: 0.9293 - val_loss: 0.0983 - val_acc: 0.8878
*****j=  96
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0586 - acc: 0.9303 - val_loss: 0.1049 - val_acc: 0.8837
*****j=  97
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0592 - acc: 0.9281 - val_loss: 0.1003 - val_acc: 0.8894
*****j=  98
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0576 - acc: 0.9310 - val_loss: 0.1040 - val_acc: 0.8839
*****j=  99
Epoch 1/1
391/391 [==============================] - 80s 204ms/step - loss: 0.0563 - acc: 0.9331 - val_loss: 0.1001 - val_acc: 0.8882


"""
