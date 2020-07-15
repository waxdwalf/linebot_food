import tensorflow as tf
import matplotlib.image as img
import numpy as np
from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import models
import cv2
from numba import cuda
from keras.utils import multi_gpu_model
from numba import jit
import math

# print(tf.__version__)
# print(tf.test.gpu_device_name())
# 根據food101檔案，將圖片分為訓練和測試
def prepare_data(filepath, src, dest):
    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')

    for food in classes_images.keys():
        print("\nCopying images into ", food)
        if not os.path.exists(os.path.join(dest, food)):
            os.makedirs(os.path.join(dest, food))
        for i in classes_images[food]:
            copy(os.path.join(src, food, i), os.path.join(dest, food, i))
    print("Copying Done!")


# 將food101中，需要的資料另外創建到 train_mini and test_mini data
def dataset_mini(food_list, src, dest):
    if os.path.exists(dest):
        rmtree(
            dest)  # removing dataset_mini(if it already exists) folders so that we will have only the classes that we want
    os.makedirs(dest)
    for food_item in food_list:
        print("Copying images into", food_item)
        copytree(os.path.join(src, food_item), os.path.join(dest, food_item))

@jit
def FinetuneInceptionv3():
    # tf.compat.v1.Session()
    K.clear_session()
    img_width, img_height = 299, 299
    train_data_dir = 'train_mini'
    validation_data_dir = 'test_mini'
    nb_train_samples = 6345  # 75750
    nb_validation_samples = 2231  # 25250
    batch_size = 18
    # 資料預處理
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # 圖片縮放比
        shear_range=0.2,  # 圖片轉換角度
        zoom_range=0.2,  # 隨機縮放幅度
        horizontal_flip=True  # 水平翻轉
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,  # 每次的資料個數
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    # 使用InceptionV3建構基礎模型
    # include_top：是否保留頂部的全連接網絡
    # weights：None代表隨機初始化，即不加載預訓練權重 imagenet代表載預訓練權重
    inception = InceptionV3(weights='imagenet', include_top=False)
    # 新增新的層數
    x = inception.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(9, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

    model = Model(inputs=inception.input, outputs=predictions)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='best_model_9class.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('history_9class.log')

    history_9class = model.fit_generator(train_generator,
                                          steps_per_epoch=math.floor(nb_train_samples / batch_size),
                                          validation_data=validation_generator,
                                          validation_steps=math.floor(nb_validation_samples / batch_size),
                                          epochs=30,
                                          verbose=1,
                                          callbacks=[csv_logger, checkpointer])

    model.save('model_trained_9class.hdf5')
    print(train_generator.class_indices)

    history = history_9class
    title = 'FOOD101-Inceptionv3'
    plt.title(title)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()

    plt.title(title)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()


def plot_accuracy(history, title):
    plt.title(title)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()


def plot_loss(history, title):
    plt.title(title)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()


# 直接使用food-101整個數據集話，需要大量時間和計算
# 因此創建了train_min和test_mini，將要得數據集限制到我們要的為9個類
food_list = ['chicken_wings', 'fried_rice', 'club_sandwich', 'caesar_salad', 'chicken_curry',
             'gyoza', 'ramen', 'Banana', 'Noodles']
src_train = 'train'
dest_train = 'train_mini'
src_test = 'test'
dest_test = 'test_mini'


def main():
    # cuda.select_device(0)
    # 根據food-101所分的train & test的資訊，進行複製資料
    # prepare_data('food-101/meta/train.txt', 'food-101/images', 'train')
    # prepare_data('food-101/meta/test.txt',  'food-101/images', 'test')
    # 複製food101要得資料
    # dataset_mini(food_list, src_train, dest_train)
    # dataset_mini(food_list, src_test, dest_test)
    # 訓練模型
    # with tf.device('/device:XLA_GPU:0'):
    FinetuneInceptionv3()



    # 測試模型
    # TestModel()


if __name__ == '__main__':
    cuda.select_device(0)
    main()