from collections import defaultdict

k = open('results_20130124.token','r')

imageNumber = {}
for line in k:
    content = line.split('#')
    actualText = ((((content[1].split('\t'))[1]).replace('\n','')).split('.'))[0].strip()
    imageNumber[content[0].strip()] = actualText.lower()

#print imageNumber

perCategCount = defaultdict(int)
filterImages = {}

for img in imageNumber:
    val = imageNumber[img]
    val_app = ' ' + val + ' '
    if ' dog ' in val_app or ' horse ' in val_app or ' cat ' in val_app or ' car ' in val_app or ' bike ' in val_app or ' bird ' in val_app or ' person ' in val_app:
        filterImages[img] = val
        if ' dog ' in val_app:
            perCategCount['dog'] +=1
        elif ' horse ' in val_app:
            perCategCount['horse'] +=1
        elif ' car ' in val_app:
            perCategCount['car'] +=1
        elif ' cat ' in val_app:
            perCategCount['cat'] +=1
        elif ' bike ' in val_app:
            perCategCount['bike'] +=1
        elif ' bird ' in val_app:
            perCategCount['bird'] +=1
        elif ' person ' in val_app:
            perCategCount['person'] +=1
        
            


#print filterImages
print perCategCount
print len(filterImages)


train = {}
validate = {}
test = {}

counterDict = defaultdict(int)

for img in filterImages:
    val = filterImages[img]
    val_app =  ' ' + val + ' '
    if ' dog ' in val_app:
        if counterDict['dog'] <= (0.8)*perCategCount['dog']:
            train[img]=val
        elif counterDict['dog'] > (0.8)*perCategCount['dog'] and counterDict['dog'] <= (13.0/15.0)*perCategCount['dog']:
            validate[img] = val
        else:
            test[img] = val
        counterDict['dog'] += 1
    elif ' cat ' in val_app:
        if counterDict['cat'] <= (0.8)*perCategCount['cat']:
            train[img]=val
        elif counterDict['cat'] > (0.8)*perCategCount['cat'] and counterDict['cat'] <= (13.0/15.0)*perCategCount['cat']:
            validate[img] = val
        else:
            test[img] = val
        counterDict['cat'] += 1
    elif ' car ' in val_app:
        if counterDict['car'] <= (0.8)*perCategCount['car']:
            train[img]=val
        elif counterDict['car'] > (0.8)*perCategCount['car'] and counterDict['car'] <= (13.0/15.0)*perCategCount['car']:
            validate[img] = val
        else:
            test[img] = val
        counterDict['car'] += 1
    elif ' horse ' in val_app:
        if counterDict['horse'] <= (0.8)*perCategCount['horse']:
            train[img]=val
        elif counterDict['horse'] > (0.8)*perCategCount['horse'] and counterDict['horse'] <= (13.0/15.0)*perCategCount['horse']:
            validate[img] = val
        else:
            test[img] = val
        counterDict['horse'] += 1
    elif ' person ' in val_app:
        if counterDict['person'] <= (0.8)*perCategCount['person']:
            train[img]=val
        elif counterDict['person'] > (0.8)*perCategCount['person'] and counterDict['person'] <= (13.0/15.0)*perCategCount['person']:
            validate[img] = val
        else:
            test[img] = val
        counterDict['person'] += 1
    elif ' bird ' in val_app:
        if counterDict['bird'] <= (0.8)*perCategCount['bird']:
            train[img]=val
        elif counterDict['bird'] > (0.8)*perCategCount['bird'] and counterDict['bird'] <= (13.0/15.0)*perCategCount['bird']:
            validate[img] = val
        else:
            test[img] = val
        counterDict['bird'] += 1
    elif ' bike ' in val_app:
        if counterDict['bike'] <= (0.8)*perCategCount['bike']:
            train[img]=val
        elif counterDict['bike'] > (0.8)*perCategCount['bike'] and counterDict['bike'] <= (13.0/15.0)*perCategCount['bike']:
            validate[img] = val
        else:
            test[img] = val
        counterDict['bike'] += 1


import glob, os, shutil

updateImgPaths = {}
for key in train:
    updateImgPaths["/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/flickr30k_images/"+key] = train[key]

files = glob.iglob(os.path.join("/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/flickr30k_images/", "*.jpg"))
for file in files:
    if file in updateImgPaths:
        shutil.copy2(file, "/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/samples/train/")


updateImgPaths = {}
for key in validate:
    updateImgPaths["/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/flickr30k_images/"+key] = validate[key]

files = glob.iglob(os.path.join("/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/flickr30k_images/", "*.jpg"))
for file in files:
    if file in updateImgPaths:
        shutil.copy2(file, "/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/samples/validate/")


updateImgPaths = {}
for key in test:
    updateImgPaths["/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/flickr30k_images/"+key] = test[key]

files = glob.iglob(os.path.join("/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/flickr30k_images/", "*.jpg"))
for file in files:
    if file in updateImgPaths:
        shutil.copy2(file, "/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/samples/test/")




import numpy as np

np.save('img_to_caption_train.npy', train)
np.save('img_to_caption_validate.npy', validate)
np.save('img_to_caption_test.npy', test)


import Image
width = 256
height = 256
files = glob.iglob(os.path.join("/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/samples/train/", "*.jpg"))
for file in files:
    fileName = file.split("/")[10]
    openImg = Image.open(file)
    openImgResize = openImg.resize((width, height), Image.NEAREST)
    openImgResize.save("/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/samplesResized/train/"+fileName)

files = glob.iglob(os.path.join("/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/samples/validate/", "*.jpg"))
for file in files:
    fileName = file.split("/")[10]
    openImg = Image.open(file)
    openImgResize = openImg.resize((width, height), Image.NEAREST)
    openImgResize.save("/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/samplesResized/validate/"+fileName)


files = glob.iglob(os.path.join("/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/samples/test/", "*.jpg"))
for file in files:
    fileName = file.split("/")[10]
    openImg = Image.open(file)
    openImgResize = openImg.resize((width, height), Image.NEAREST)
    openImgResize.save("/home/utkarsh1404/Documents/CS682/Project/dataset/flickr30k_images/samplesResized/test/"+fileName)


