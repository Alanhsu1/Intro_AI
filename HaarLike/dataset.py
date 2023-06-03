import os
import tensorflow as tf
import numpy as np

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    dataset = list()
    for pt in os.listdir(dataPath):
        for fn in os.listdir(dataPath+'/'+pt):
            img = tf.keras.utils.load_img(dataPath+'/'+pt+'/'+fn, target_size=(24, 24))

            img_tmp = tf.keras.utils.img_to_array(img)
            img_tmp = tf.image.rgb_to_grayscale(img_tmp)
            # float32 轉 int64
            img_tmp = tf.cast(img_tmp, tf.int64)
            # img_tmp shape is (24,24,1)
            # type <class 'tensorflow.python.framework.ops.EagerTensor'>
            
            img_final = tf.squeeze(img_tmp)
            # img_final shape is (24,24)

            if pt=='COVID': # label is 1 when it is a COVID
                dataset.append((img_final, 1))
            else:
                dataset.append((img_final, 0))
    return dataset
    # dataset is a list of tuples

dt = loadImages('CT')