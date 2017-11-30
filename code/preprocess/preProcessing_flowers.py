import gensim
import numpy as np
model = gensim.models.KeyedVectors.load_word2vec_format('/home/utkarsh1404/GoogleNews-vectors-negative300.bin', binary=True)

#print model['tree']
#print model['Tree']
#print model['and']


sunflower = np.reshape(np.array(model['sunflower']),(1,300))
gazania = np.reshape(np.array(model['gazania']),(1,300))

train_vec = {}
validate_vec = {}
test_vec = {}

import glob, os, shutil
files = glob.iglob(os.path.join("/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/train/", "*.jpg"))
for file in files:
    fileName = file.split("/")[8]
    print fileName
    if 'sunflower' in fileName:
        train_vec[fileName] = np.copy(sunflower)
    else:
        train_vec[fileName] = np.copy(gazania)

files = glob.iglob(os.path.join("/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/validate/", "*.jpg"))
for file in files:
    fileName = file.split("/")[8]
    if 'sunflower' in fileName:
        validate_vec[fileName] = np.copy(sunflower)
    else:
        validate_vec[fileName] = np.copy(gazania)

files = glob.iglob(os.path.join("/home/utkarsh1404/project/text2image/data/flowers/flowerSamplesResized/test/", "*.jpg"))
for file in files:
    fileName = file.split("/")[8]
    if 'sunflower' in fileName:
        test_vec[fileName] = np.copy(sunflower)
    else:
        test_vec[fileName] = np.copy(gazania)


np.save('/home/utkarsh1404/project/text2image/data/flowers/system_input_train.npy', train_vec)
np.save('/home/utkarsh1404/project/text2image/data/flowers/system_input_validate.npy', validate_vec)
np.save('/home/utkarsh1404/project/text2image/data/flowers/system_input_test.npy', test_vec)

print train_vec
