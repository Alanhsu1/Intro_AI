import tensorflow as tf
from os import listdir
from os.path import join, isfile
import numpy as np

# Modify by yourself
img_path = 'C:/Users/yifan/Desktop/Intro_AI/CT/COVID'
img_file = np.array([join(img_path, f) for f in listdir(img_path) if isfile(join(img_path, f))])
X = np.array([])

cnt = 0
for file in img_file:
    cnt += 1
    if cnt == 100:
        break
    img = tf.keras.utils.load_img(file, target_size=(224, 224))

    img2 = tf.keras.utils.img_to_array(img)
    img2 = tf.image.rgb_to_grayscale(img2)

    img2 = np.expand_dims(img2, axis=0)
    # print(np.shape(img2))

    if len(X.shape) == 1:
        X = img2
    else:
        X = np.concatenate((X, img2), axis=0)

# print(X[0])