import gensim
import numpy as np
import os
cwd = os.getcwd()
#model = gensim.models.KeyedVectors.load_word2vec_format('/home/utkarsh1404/Documents/CS682/Project/dataset/GoogleNews-vectors-negative300.bin', binary=True)

#print model['tree']
#print model['Tree']
#print model['and']

train = np.load(cwd+'/../../../flickr30k_images/img_to_caption_train.npy').item()
validate = np.load(cwd+'/../../../flickr30k_images/img_to_caption_validate.npy').item()
test = np.load(cwd+'/../../../flickr30k_images/img_to_caption_test.npy').item()


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import string

nltk.download('stopwords')
nltk.download('punkt')
 
stopWords = set(stopwords.words('english'))

maxLen = -1
sent = []
for key in train:
    val = train[key]
    words = word_tokenize(val)
    wordsFiltered = []
    for w in words:
        if w not in stopWords and w not in string.punctuation:
            wordsFiltered.append(w)
 
    currlen = len(wordsFiltered)
    if currlen > maxLen:
        maxLen = currlen
        sent = wordsFiltered
for key in validate:
    val = validate[key]
    words = word_tokenize(val)
    wordsFiltered = []
    for w in words:
        if w not in stopWords and w not in string.punctuation:
            wordsFiltered.append(w)
 
    currlen = len(wordsFiltered)
    if currlen > maxLen:
        maxLen = currlen
        sent = wordsFiltered
for key in test:
    val = test[key]
    words = word_tokenize(val)
    wordsFiltered = []
    for w in words:
        if w not in stopWords and w not in string.punctuation:
            wordsFiltered.append(w)
 
    currlen = len(wordsFiltered)
    if currlen > maxLen:
        maxLen = currlen
        sent = wordsFiltered

print maxLen
print sent
