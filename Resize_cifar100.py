import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
import keras
from keras.datasets import cifar10,cifar100
from keras.models import Model
from keras.utils import to_categorical
from keras.models import Sequential
from keras import initializers, layers
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import layers, models, optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
import os
from keras import regularizers
import cv2
from PIL import Image

if __name__ == '__main__':
    # CIFAR-10データセットをロード
    (X_train1, Y_train), (X_test1, Y_test) = cifar100.load_data()
    print(X_train1.shape, Y_train.shape)
    print(X_test1.shape, Y_test.shape)

    X_train1 = np.array(X_train1)[:50000]  #/ 255
    Y_train = np.array(Y_train)[:50000].astype(np.int32)
    X_test1 = np.array(X_test1)[:10000] #/ 255
    Y_test = np.array(Y_test)[:10000].astype(np.int32)
    img_rows, img_cols = 128, 128
    X_train =[]
    X_test = []
    path ='E:\Cifar100_128'  #E:\Cifar100_128 #./Cifar100_128/
    for i in range(50000):
        dst = cv2.resize(X_train1[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC) #cv2.INTER_LINEAR #cv2.INTER_CUBIC
        dst = dst[:,:,::-1]  #追記；理由はおまけに記載
        #img=Image.open(path+'/X_train/'+str(i)+'.jpg')
        #dst = img.resize(img_rows, img_cols)
        #print(i,str(Y_train[i][0]))
        X_train.append(dst)
        cv2.imwrite(path+'/X_train_128/'+str(Y_train[i][0])+'/'+str(i)+'.jpg', dst)
    for i in range(10000):
        dst = cv2.resize(X_test1[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
        dst = dst[:,:,::-1]  #追記；理由はおまけに記載
        #img=Image.open(path+'/X_test/'+str(i)+'.jpg')
        #dst = img.resize(img_rows, img_cols)
        X_test.append(dst)
        cv2.imwrite(path+'/X_test_128/'+str(Y_test[i][0])+'/'+str(i)+'.jpg', dst)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    """
    # 画像を描画
    nclasses = 100
    pos = 1
    for targetClass in range(nclasses):
        targetIdx = []
        # クラスclassIDの画像のインデックスリストを取得
        for i in range(len(Y_train)):
            if Y_train[i][0] == targetClass:
                targetIdx.append(i)

        # 各クラスからランダムに選んだ最初の10個の画像を描画
        #np.random.shuffle(targetIdx)
        plt.figure(figsize=(10,10))
        for idx in targetIdx[:10]:
            img = toimage(X_train[idx])
            plt.subplot(10, 10, pos)
            plt.imshow(img)
            plt.axis('off')
            pos += 1
    plt.savefig("./caps_figures/resize_cifar10.png")
    plt.pause(3)
    plt.close()
    """