import numpy as np
import os
from PIL import Image


pixelSz = 32

def imgtobin_dogs(tanh_flag):
    
    if tanh_flag==0:
        train_dict = {}
        val_dict = {}
        test_dict = {}

        sz = 0
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/train'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    sz+=1

        dummy_arr = np.zeros((sz, pixelSz, pixelSz, 3))
        ct=0
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/train'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    dummy_arr[ct] = pix
                    ct+=1

        mean = np.mean(dummy_arr, axis=0)
        std = np.std(dummy_arr, axis=0)

        print mean.shape


        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/train'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix - mean)/(1.0*std)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    train_dict[filename] = pix
         
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/validate'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix - mean)/(1.0*std)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    val_dict[filename] = pix
           
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/test'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix - mean)/(1.0*std)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    test_dict[filename] = pix

        return train_dict, val_dict, test_dict
    elif tanh_flag==1:
        train_dict = {}
        val_dict = {}
        test_dict = {}
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/train'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix)/(255.0)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    train_dict[filename] = pix
         
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/validate'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix)/(255.0)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    val_dict[filename] = pix
           
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/test'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix)/(255.0)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    test_dict[filename] = pix

        return train_dict, val_dict, test_dict
    elif tanh_flag==2:
        train_dict = {}
        val_dict = {}
        test_dict = {}
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/train'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    w, h, c = pix.shape
                    maxVal = np.max(pix, axis=(0,1))*1.0
                    minVal = np.min(pix, axis=(0,1))*1.0
                    for id in range(c):
                        currMinVal = minVal[id]
                        currMaxVal = maxVal[id]
                        for l1 in range(w):
                            for l2 in range(h):
                                pix[l1][l2][id] = ((pix[l1][l2][id] - currMinVal)/(currMaxVal-currMinVal))
                    pix = np.reshape(pix, (c, w, h))

                    train_dict[filename] = pix
         
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/validate'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    w, h, c = pix.shape
                    maxVal = np.max(pix, axis=(0,1))*1.0
                    minVal = np.min(pix, axis=(0,1))*1.0
                    for id in range(c):
                        currMinVal = minVal[id]
                        currMaxVal = maxVal[id]
                        for l1 in range(w):
                            for l2 in range(h):
                                pix[l1][l2][id] = ((pix[l1][l2][id] - currMinVal)/(currMaxVal-currMinVal))
                    pix = np.reshape(pix, (c, w, h))

                    val_dict[filename] = pix
           
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/dogs/samplesResized/test'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    w, h, c = pix.shape
                    maxVal = np.max(pix, axis=(0,1))*1.0
                    minVal = np.min(pix, axis=(0,1))*1.0
                    for id in range(c):
                        currMinVal = minVal[id]
                        currMaxVal = maxVal[id]
                        for l1 in range(w):
                            for l2 in range(h):
                                pix[l1][l2][id] = ((pix[l1][l2][id] - currMinVal)/(currMaxVal-currMinVal))
                    pix = np.reshape(pix, (c, w, h))

                    test_dict[filename] = pix

        return train_dict, val_dict, test_dict
        

def imgtobin_flowers(tanh_flag):
    
    if tanh_flag==0:
        train_dict = {}
        val_dict = {}
        test_dict = {}

        sz = 0
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/train'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    sz+=1

        dummy_arr = np.zeros((sz, pixelSz, pixelSz, 3))
        ct=0
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/train'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    dummy_arr[ct] = pix
                    ct+=1

        mean = np.mean(dummy_arr, axis=0)
        std = np.std(dummy_arr, axis=0)

        print mean.shape


        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/train'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix - mean)/(1.0*std)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    train_dict[filename] = pix
         
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/validate'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix - mean)/(1.0*std)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    val_dict[filename] = pix
           
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/test'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix - mean)/(1.0*std)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    test_dict[filename] = pix

        return train_dict, val_dict, test_dict
    elif tanh_flag==1:
        train_dict = {}
        val_dict = {}
        test_dict = {}
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/train'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix)/(255.0)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    train_dict[filename] = pix
         
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/validate'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix)/(255.0)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    val_dict[filename] = pix
           
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/test'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    pix = (pix)/(255.0)
                    #print(os.path.join(dirname, filename))

                    w, h, c = pix.shape 
                    pix = np.reshape(pix, (c, w, h))

                    test_dict[filename] = pix

        return train_dict, val_dict, test_dict
    elif tanh_flag==2:
        train_dict = {}
        val_dict = {}
        test_dict = {}
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/train'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    w, h, c = pix.shape
                    maxVal = np.max(pix, axis=(0,1))*1.0
                    minVal = np.min(pix, axis=(0,1))*1.0
                    for id in range(c):
                        currMinVal = minVal[id]
                        currMaxVal = maxVal[id]
                        for l1 in range(w):
                            for l2 in range(h):
                                pix[l1][l2][id] = ((pix[l1][l2][id] - currMinVal)/(currMaxVal-currMinVal))
                    pix = np.reshape(pix, (c, w, h))

                    train_dict[filename] = pix
         
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/validate'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    w, h, c = pix.shape
                    maxVal = np.max(pix, axis=(0,1))*1.0
                    minVal = np.min(pix, axis=(0,1))*1.0
                    for id in range(c):
                        currMinVal = minVal[id]
                        currMaxVal = maxVal[id]
                        for l1 in range(w):
                            for l2 in range(h):
                                pix[l1][l2][id] = ((pix[l1][l2][id] - currMinVal)/(currMaxVal-currMinVal))
                    pix = np.reshape(pix, (c, w, h))

                    val_dict[filename] = pix
           
        for dirname, dirnames, filenames in os.walk('/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/test'):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    pix = Image.open(os.path.join(dirname, filename))
                    pix = np.array(pix, dtype=np.float)
                    w, h, c = pix.shape
                    maxVal = np.max(pix, axis=(0,1))*1.0
                    minVal = np.min(pix, axis=(0,1))*1.0
                    for id in range(c):
                        currMinVal = minVal[id]
                        currMaxVal = maxVal[id]
                        for l1 in range(w):
                            for l2 in range(h):
                                pix[l1][l2][id] = ((pix[l1][l2][id] - currMinVal)/(currMaxVal-currMinVal))
                    pix = np.reshape(pix, (c, w, h))

                    test_dict[filename] = pix

        return train_dict, val_dict, test_dict
        
