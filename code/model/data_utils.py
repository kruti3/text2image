import array 
import cPickle as pickle
import numpy as np
import os
from PIL import Image
from array import *

def imgtobin():
    
    train_dict = {}
    val_dict = {}
    test_dict = {}

    for dirname, dirnames, filenames in os.walk('../../data/samplesResized/train'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                im = Image.open(os.path.join(dirname, filename))
                pix = im.load()
                pix = np.array(pix)
                #print(os.path.join(dirname, filename))

                w, h, c = pix.shape 
                pix = np.reshape(pix, (c, w, h))

                train_dict[filename] = pix
     
    for dirname, dirnames, filenames in os.walk('../../data/samplesResized/validate'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                im = Image.open(os.path.join(dirname, filename))
                pix = im.load()
                pix = np.array(pix)
                #print(os.path.join(dirname, filename))

                w, h, c = pix.shape 
                pix = np.reshape(pix, (c, w, h))

                val_dict[filename] = pix
       
    for dirname, dirnames, filenames in os.walk('../../data/samplesResized/test'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                im = Image.open(os.path.join(dirname, filename))
                pix = im.load()
                pix = np.array(pix)
                #print(os.path.join(dirname, filename))

                w, h, c = pix.shape 
                pix = np.reshape(pix, (c, w, h))

                test_dict[filename] = pix

