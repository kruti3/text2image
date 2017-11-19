import gensim
import numpy as np
model = gensim.models.KeyedVectors.load_word2vec_format('/home/kruti/text2image/GoogleNews-vectors-negative300.bin', binary=True)

#print model['tree']
#print model['Tree']
#print model['and']

train = np.load('/home/kruti/text2image/data/img_to_caption_train.npy').item()
validate = np.load('/home/kruti/text2image/data/img_to_caption_validate.npy').item()
test = np.load('/home/kruti/text2image/data/img_to_caption_test.npy').item()


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import string

nltk.download('stopwords')
nltk.download('punkt')
 
stopWords = set(stopwords.words('english'))


final_train = {}
final_validate = {}
final_test = {}

maxLen = -1
sent = []
for key in train:
    val = train[key]
    words = word_tokenize(val)
    wordsFiltered = []
    for w in words:
        if w not in stopWords and w not in string.punctuation:
            wordsFiltered.append(w)
    final_train[key] = wordsFiltered
 
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
    final_validate[key] = wordsFiltered
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
    final_test[key] = wordsFiltered
    currlen = len(wordsFiltered)
    if currlen > maxLen:
        maxLen = currlen
        sent = wordsFiltered

print maxLen
print sent


train_vec = {}
validate_vec = {}
test_vec = {}

for key in final_train:
    currArray = np.zeros((11,300))
    words = final_train[key]
    counter = 0
    for word in words:
        upperWord = word[0].upper() + word[1:]
        if word in model:
            vec = model[word]
            for i in range(0,300):
                currArray[counter][i] = vec[i]
        elif upperWord in model:
            vec = model[upperWord]
            for i in range(0,300):
                currArray[counter][i] = vec[i]
        counter+=1
    train_vec[key] = currArray   

for key in final_validate:
    currArray = np.zeros((11,300))
    words = final_validate[key]
    counter = 0
    for word in words:
        upperWord = word[0].upper() + word[1:]
        if word in model:
            vec = model[word]
            for i in range(0,300):
                currArray[counter][i] = vec[i]
        elif upperWord in model:
            vec = model[upperWord]
            for i in range(0,300):
                currArray[counter][i] = vec[i]
        counter+=1
    validate_vec[key] = currArray   

for key in final_test:
    currArray = np.zeros((11,300))
    words = final_test[key]
    counter = 0
    for word in words:
        upperWord = word[0].upper() + word[1:]
        if word in model:
            vec = model[word]
            for i in range(0,300):
                currArray[counter][i] = vec[i]
        elif upperWord in model:
            vec = model[upperWord]
            for i in range(0,300):
                currArray[counter][i] = vec[i]
        counter+=1
    test_vec[key] = currArray   


np.save('/home/kruti/text2image/data/system_input_train.npy', train_vec)
np.save('/home/kruti/text2image/data/system_input_validate.npy', validate_vec)
np.save('/home/kruti/text2image/data/system_input_test.npy', test_vec)

print validate_vec
