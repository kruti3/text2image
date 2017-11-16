import array 
import cPickle as pickle
import numpy as np
import os
from PIL import Image
from array import *

def imgtobin(name):
    
    data = array('B')
 
    for dirname, dirnames, filenames in os.walk('../../data/samplesResized/' + name):
        for filename in filenames:
            if filename.endswith('.jpg'):

                im = Image.open(os.path.join(dirname, filename))
                im = np.array(im)
                print(os.path.join(dirname, filename))
            
                for color in range(0,3):
                    for x in range(0,256):
                        for y in range(0,256):
                            data.append(im[x,y][color])

    output_file = open('../../data/' + name, 'wb')
    data.tofile(output_file)
    output_file.close()


if __name__ == '__main__':

    imgtobin('train')
    imgtobin('test')
    imgtobin('validate')
