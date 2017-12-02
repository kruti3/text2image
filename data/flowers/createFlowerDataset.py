from collections import defaultdict
import scipy.io as sio
import os

labelsD = sio.loadmat('imagelabels.mat')['labels']

labels = []
for key in labelsD:
    for k in key:
        labels.append(k)



storelabel = []
fileNames = []

fileNameToLabel = {}

for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/jpg'):
    for filename in filenames:
        if filename.endswith('.jpg'):
            fileNames.append(os.path.join(dirname, filename))


fileNames.sort()
print len(labels)
print len(fileNames)


ct=0
for filename in fileNames:
    #print fileName
    if 'image_05399.jpg'  in filename or 'image_04481.jpg'  in filename or 'image_01314.jpg'  in filename or 'image_01696.jpg'  in filename or labels[ct] in storelabel:
        storelabel.append(labels[ct])
        if '05399' in filename: 
            print labels[ct], "=", "sunflower"
        elif '04481' in filename:
            print labels[ct], "=", "garbanza"
        elif '01696' in filename:
            print labels[ct], "=", "hibiscus"
        elif '01314' in filename:
            print labels[ct], "=", "petunia"
        fileNameToLabel[filename] = str(labels[ct])
    ct+=1


print storelabel
print len(storelabel)
print set(storelabel)
#print fileNameToLabel

d = {}
d['54'] = 'sunflower'
d['71'] = 'gazania'
d['51'] = 'petunia'
d['83'] = 'hibiscus'



import glob, os, shutil

files = glob.iglob(os.path.join("/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/jpg/", "*.jpg"))
for file in files:
    if file in fileNameToLabel.keys():
        shutil.copy2(file, "/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/flowerSamples/train/")
        shutil.copy2(file, "/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/flowerSamples/test/")
        shutil.copy2(file, "/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/flowerSamples/validate/")

import numpy as np
import Image

width = 32
height = 32
files = glob.iglob(os.path.join("/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/flowerSamples/train/", "*.jpg"))
for file in files:
    fileName = file.split("/")[10]
    orgFileN = fileName
    fileName = (fileName.replace('.jpg','')) +'___'+ d[fileNameToLabel['/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/jpg/'+orgFileN]] + ".jpg"
    openImg = Image.open(file)
    openImgResize = openImg.resize((width, height), Image.NEAREST)
    openImgResize.save("/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/flowerSamplesResized/train/"+fileName)

files = glob.iglob(os.path.join("/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/flowerSamples/validate/", "*.jpg"))
for file in files:
    fileName = file.split("/")[10]
    orgFileN = fileName
    fileName = (fileName.replace('.jpg','')) +'___'+ d[fileNameToLabel['/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/jpg/'+orgFileN]] + ".jpg"
    openImg = Image.open(file)
    openImgResize = openImg.resize((width, height), Image.NEAREST)
    openImgResize.save("/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/flowerSamplesResized/validate/"+fileName)


files = glob.iglob(os.path.join("/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/flowerSamples/test/", "*.jpg"))
for file in files:
    fileName = file.split("/")[10]
    orgFileN = fileName
    fileName = (fileName.replace('.jpg','')) +'___'+ d[fileNameToLabel['/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/jpg/'+orgFileN]] + ".jpg"
    openImg = Image.open(file)
    openImgResize = openImg.resize((width, height), Image.NEAREST)
    openImgResize.save("/home/utkarsh1404/Documents/CS682/Project/dataset/flowers/flowerSamplesResized/test/"+fileName)
