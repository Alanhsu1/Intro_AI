from configparser import Interpolation
import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    f = open(dataPath, 'r')
    while True: # keep going until EOF
        rd=f.readline()
        if rd=='': break #EOF comes

        nm, cnt = rd.split()
        
        img = cv2.imread('data/detect/'+str(nm))
        gray = cv2.imread('data/detect/'+str(nm), cv2.IMREAD_GRAYSCALE)
        for c in range(int(cnt)):
            x, y, wid, hei = list(map(int, f.readline().split()))
            crop = gray[y:y+hei, x:x+wid]

            rsz = cv2.resize(crop, (19, 19), interpolation=cv2.INTER_NEAREST)

            if clf.classify(rsz)==1: #green
                cv2.rectangle(img, (x,y), (x+wid,y+hei), (0,255,0), 3)
            else: #red
                cv2.rectangle(img, (x,y), (x+wid,y+hei), (0,0,255), 3)
                
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    f.close()
    # End your code (Part 4)
