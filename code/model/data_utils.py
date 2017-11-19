import numpy as np
import os
from PIL import Image

def imgtobin():
    
    train_dict = {}
    val_dict = {}
    test_dict = {}

    sz = 0
    for dirname, dirnames, filenames in os.walk('/home/kruti/text2image/data/samplesResized/train'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                sz+=1

    dummy_arr = np.zeros((sz, 128, 128, 3))
    ct=0
    for dirname, dirnames, filenames in os.walk('/home/kruti/text2image/data/samplesResized/train'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                pix = Image.open(os.path.join(dirname, filename))
                pix = np.array(pix, dtype=np.float)
                dummy_arr[ct] = pix
                ct+=1

    mean = np.mean(dummy_arr, axis=0)
    std = np.std(dummy_arr, axis=0)

    print mean.shape


    for dirname, dirnames, filenames in os.walk('/home/kruti/text2image/data/samplesResized/train'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                pix = Image.open(os.path.join(dirname, filename))
                pix = np.array(pix, dtype=np.float)
                pix = (pix - mean)/(1.0*std)
                #print(os.path.join(dirname, filename))

                w, h, c = pix.shape 
                pix = np.reshape(pix, (c, w, h))

                train_dict[filename] = pix
     
    for dirname, dirnames, filenames in os.walk('/home/kruti/text2image/data/samplesResized/validate'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                pix = Image.open(os.path.join(dirname, filename))
                pix = np.array(pix, dtype=np.float)
                pix = (pix - mean)/(1.0*std)
                #print(os.path.join(dirname, filename))

                w, h, c = pix.shape 
                pix = np.reshape(pix, (c, w, h))

                val_dict[filename] = pix
       
    for dirname, dirnames, filenames in os.walk('/home/kruti/text2image/data/samplesResized/test'):
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

