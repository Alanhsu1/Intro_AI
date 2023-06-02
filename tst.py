import tensorflow as tf
from os import listdir
from os.path import isfile, join
from os import path
import numpy as np

img_path = './CT/'
image_files = np.array([join(img_path, f) for f in listdir(img_path) if isfile(join(img_path, f))])
X = np.array([])
# 讀取 data/without_mask 目錄下所有圖檔
for f in image_files:
    # 載入圖檔，並縮放寬高為 (224, 224) 
    img = tf.keras.utils.load_img(f, target_size=(224, 224))
    # 加一維，變成 (1, 224, 224, 3)，最後一維是色彩
    img2 = tf.keras.utils.img_to_array(img)
    img2 = np.expand_dims(img2, axis=0)
    if len(X.shape) == 1:
        X = img2
    else:
        # 合併每個圖檔的像素
        X = np.concatenate((X, img2), axis=0)
    print(X.shape)
# dt = tf.keras.utils.image_dataset_from_directory('./CT/')

# for i in dt.as_numpy_iterator():
#     print(type(i))
#     print(len(i))
#     print(len(i[0]))
