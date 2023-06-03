import os
import cv2
import torchvision.transforms as trns
from os import listdir
from os.path import join, isfile
from PIL import Image


def loadImages(dataPath):
    img_path = dataPath + '/COVID'
    # resize the picture into 224 * 224 
    transform = trns.Compose([trns.Resize((224, 224)), 
                              trns.ToTensor(), 
                              trns.Normalize(
                                mean=[0.485, 0.456, 0.406], 
		                        std=[0.229, 0.224, 0.225])])

    img_file = [join(img_path, f) for f in listdir(img_path) if isfile(join(img_path, f))]
    # print(img_file)

    dataset = []
    cnt = 0
    for file in img_file:
        cnt += 1
        if cnt == 100:
            break
        img = Image.open(file).convert("RGB")
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        
        dataset.append(img_tensor)

    return dataset

if __name__ == "__main__":
    # loadImages
    pass