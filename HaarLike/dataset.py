import os
import cv2
import tensorflow as tf
import numpy as np
from os import listdir
from os.path import join, isfile

def loadImages(dataPath):
    # datapath = './CT'
    dataset = []
    dataPath1 = dataPath + "/COVID"
    img_file = np.array([join(dataPath1, f) for f in listdir(dataPath1) if isfile(join(dataPath1, f))])
    cnt = 0
    
    for file in img_file:
        cnt += 1
        if cnt == 2:
            break
        img = tf.keras.utils.load_img(file, target_size=(224, 224))

        img_tmp = tf.keras.utils.img_to_array(img)
        img_tmp = tf.image.rgb_to_grayscale(img_tmp)
        img_tmp = tf.cast(img_tmp, tf.int64)
        # print(type(img_tmp))
        img_tmp = np.expand_dims(img_tmp, axis=0)
        print(img_tmp)
        if cnt <= 100:
            dataset.append((img_tmp, 1))
        else:
            dataset.append((img_tmp, 1))

    return dataset
