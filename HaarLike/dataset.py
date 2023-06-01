import os
import cv2

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
    # Begin your code (Part 1)
    dataset = list()
    for pt in os.listdir(dataPath):
        for fn in os.listdir(dataPath+'/'+pt):
            if pt=='face': # label is 1 when it is a face
                tup=(cv2.imread(dataPath+'/'+pt+'/'+fn, cv2.IMREAD_GRAYSCALE), 1)
            else:
                tup=(cv2.imread(dataPath+'/'+pt+'/'+fn, cv2.IMREAD_GRAYSCALE), 0)
            dataset.append(tup)
    # End your code (Part 1)
    return dataset
