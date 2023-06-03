import os
import random
import shutil
filedir = "./CT/COVID/"
tardir = "./CT/test/COVID/"
pathdir = os.listdir(filedir)
num = len(pathdir)
pick = int(num * 0.3)
sample = random.sample(pathdir, pick)
print(sample)
for name in sample:
    shutil.move(filedir + name, tardir + name)