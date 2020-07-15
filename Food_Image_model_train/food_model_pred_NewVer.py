
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import glob
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import os
from keras.models import load_model

import tensorflow as tf
from tensorflow import keras
import h5py
#將圖片規格化
def read_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(299, 299))
    except Exception as e:
        print(img_path,e)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img/255

#製作辨識圖片
def draw_save(img_path, label, out='tmp/'):
    img = Image.open(img_path)
    os.makedirs(os.path.join(out,label),exist_ok=True)
    if img is None:return None
    # 在圖片上加入文字
    draw = ImageDraw.Draw(img)
    # 使用中文字形
    font = ImageFont.truetype("TW-Kai-98_1.ttf",160)
    #fill文字顏色 黃色
    draw.text((10,10), label, fill='#FFFF00', font=font)
    #將辨識好的圖片存檔
    img.save(os.path.join(out,label,"test.jpg"))

#顯示分類的名稱術與數量
train_dir = 'train_mini'
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir)
index = train_generator.class_indices
print('index:', index)
labels = index
labels = {str(v): k for k, v in labels.items()}
#將上面index內的東西複製並形成字典
# labels = {'Fried_noodles': 0, 'Japanese_noodle': 1, 'banana': 2, 'caesar_salad': 3, 'chicken_wings': 4, 'curry_rice': 5, 'dumplings': 6, 'fried_rice': 7, 'sandwich': 8}
# labels = {str(v): k for k, v in labels.items()}
# print(labels)

# model = load_model('model_trained_9class.hdf5') #導入辨識模型
model = keras.models.load_model('model_trained_9class20200622.hdf5')

# # 隨選擇欲辨識的圖片所在資料夾
# files = glob.glob("tmp/*.jpg")
#
# #選擇要辨識的圖片編號，若不清楚圖片名稱與ID的對應關係，請用下方的print查詢
# testID = 0
# print('testID:', files[testID])


for file in os.listdir("tmp/"):

    img = read_image('tmp/'+file)
    pred_row = model.predict(img)
    pred = model.predict(img)[0]
    # print(pred)
    # print('pred_row', pred_row)
    # print('pred_row_type', type(pred_row))


    #推論出機率最高的分類, 取得所在位置
    index = np.argmax(pred)
    if pred[index] > 0.9:
        print(file, '   與  ', labels[str(index)],'  相似度為:   ', round(pred[index]*100, 2),"%")
    else:
        print(file, '\033[7;31m 我就爛!!辨識不出來!! \033[0m', '   與  ', labels[str(index)],'  相似度為:   ', round(pred[index]*100, 2),"%")
    # draw_save(files[testID], labels[str(index)], out='tmp/')