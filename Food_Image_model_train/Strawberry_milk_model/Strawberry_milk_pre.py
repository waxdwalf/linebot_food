# --coding:utf-8--
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import glob
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import os
from keras.models import load_model


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
# def draw_save(img_path, label, out='output/'):
#     img = Image.open(img_path)
#     os.makedirs(os.path.join(out,label),exist_ok=True)
#     if img is None:return None
#     # 在圖片上加入文字
#     draw = ImageDraw.Draw(img)
#     # 使用中文字形
#     font = ImageFont.truetype("TW-Kai-98_1.ttf",160)
#     #fill文字顏色 黃色
#     draw.text((10,10), label, fill='#FFFF00', font=font)
#     #將辨識好的圖片存檔
#     img.save(os.path.join(out,label,'123.jpg'))

#顯示分類的名稱術與數量
train_dir = 'Strawberry_milk_model_image/train'
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir)
index = train_generator.class_indices
print('index:', index)

#將上面index內的東西複製並形成字典
labels = {'Strawberry': 0, 'milk': 1}
labels = {str(v): k for k,v in labels.items()}
print(labels)

# 隨選擇欲辨識的圖片所在資料夾
files = glob.glob("test/*.png")

print(len(files))

files_count = int(len(files))

testID = 0

for testID in range(0, files_count):
    # 選擇要辨識的圖片編號，若不清楚圖片名稱與ID的對應關係，請用下方的print查詢

    print('testID:', files[testID])

    model = load_model('Strawberry_milk_model_250.h5') #導入辨識模型
    img = read_image(files[testID])
    pred_row = model.predict(img)
    pred = model.predict(img)[0]
    # print('pred:', pred)

    #推論出機率最高的分類, 取得所在位置
    index = np.argmax(pred)

    print(files[testID], labels[str(index)], pred[index])
    # draw_save(files[testID], labels[str(index)], out='tmp/')

    testID= testID+1
