from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
import cv2
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 200

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test_original=x_test
y_test_original=y_test
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

early_stopping = EarlyStopping(monitor='val_acc', patience=20, mode='max', verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, 
                          factor=0.5, min_lr=0.000001, verbose=1)
csv_logger = CSVLogger('history_mnist.log', separator=',', append=True)
cbks = [early_stopping, lr_reduction, csv_logger]
#cbks = [lr_reduction, csv_logger]


model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',  #RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,

                    ## add 1 line
                    callbacks=cbks,

                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

# Prediction
import numpy as np
from sklearn.metrics import confusion_matrix

predict_classes = model.predict_classes(x_test[1:10000,], batch_size=32)
true_classes = np.argmax(y_test[1:10000],1)
#print(confusion_matrix(true_classes, predict_classes))

path = './mnist'
img_rows, img_cols=112,112
print(predict_classes)

for i in range(1,10000,1):
    dst = cv2.resize(x_test_original[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    print(i, y_test_original[i],predict_classes[i-1],true_classes[i-1])
    #cv2.imwrite(path+'/X_test/'+str(int(y_test_original[i]))+'/'+str(i)+'.jpg', dst)
    if predict_classes[i-1] != true_classes[i-1]:
        plt.imshow(dst)
        plt.pause(0.1)
        plt.close()
        cv2.imwrite(path+'/X_test/'+str(int(y_test_original[i]))+'/'+str(int(predict_classes[i-1]))+'/'+str(i)+'.jpg', dst)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(confusion_matrix(true_classes, predict_classes))        