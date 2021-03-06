import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt
from PIL import Image

from data_utils import *

from lasagne.layers import *
from lasagne.nonlinearities import *

from random import randint


imageIdToNameDict = {}
imageIdToCaptionVectorDict = {}
X_train_img, X_train_caption, X_val_img, X_val_caption, X_test_img, X_test_caption = None, None, None, None, None, None


def load_dataset():
    train_img_raw, val_img_raw, test_img_raw = imgtobin()

    #print test_img_raw
    
    train_caption = np.load('/home/kruti/text2image/data/system_input_train.npy').item()
    validate_caption = np.load('/home/kruti/text2image/data/system_input_validate.npy').item()
    test_caption = np.load('/home/kruti/text2image/data/system_input_test.npy').item()

    train_sz = len(train_caption)
    X_train_caption_lcl = np.zeros((train_sz, 11 , 300))
    X_train_img_lcl = np.zeros((train_sz, 3 , 128, 128))
    validate_sz = len(validate_caption)
    X_val_caption_lcl = np.zeros((validate_sz, 11 , 300))
    X_val_img_lcl = np.zeros((validate_sz, 3 , 128, 128))
    test_sz = len(test_caption)
    X_test_caption_lcl = np.zeros((test_sz, 11 , 300))
    X_test_img_lcl = np.zeros((test_sz, 3 , 128, 128))
    
    global_ctr = 0
    counter = 0
    for key in train_caption:
        imageIdToNameDict[global_ctr] = key
        imageIdToCaptionVectorDict[global_ctr] = train_caption[key]
        X_train_caption_lcl[counter] = train_caption[key]
        X_train_img_lcl[counter] = train_img_raw[key]
        counter+=1
        global_ctr+=1
    counter = 0
    for key in validate_caption:
        imageIdToNameDict[global_ctr] = key
        imageIdToCaptionVectorDict[global_ctr] = validate_caption[key]
        X_val_caption_lcl[counter] = validate_caption[key]
        X_val_img_lcl[counter] = val_img_raw[key]
        counter+=1
        global_ctr+=1
    counter = 0
    for key in test_caption:
        imageIdToNameDict[global_ctr] = key
        imageIdToCaptionVectorDict[global_ctr] = test_caption[key]
        X_test_caption_lcl[counter] = test_caption[key]
        X_test_img_lcl[counter] = test_img_raw[key]
        counter+=1
        global_ctr+=1

    
    return X_train_img_lcl, X_train_caption_lcl, X_val_img_lcl, X_val_caption_lcl, X_test_img_lcl, X_test_caption_lcl


def get_sampled_batch_for_training(imgs, captions, batch_size):
    lst = []
    size = imgs.shape[0]
    while len(lst)!=batch_size:
        randNum = randint(0, size-1)
        if randNum not in lst:
            lst.append(randNum)

    return imgs[lst], captions[lst]


def disc_model(input_img, input_text):
    # disc layer description
    lrelu = LeakyRectify(0.1)

    input_dis = InputLayer(shape = (None, 3, 128, 128), input_var = input_img)
    frst_conv_layer = batch_norm(Conv2DLayer(input_dis, 17, 5, stride=1, pad=2, nonlinearity=lrelu))
    second_conv_layer = batch_norm(Conv2DLayer(frst_conv_layer, 11, 5, stride=1, pad=2, nonlinearity=lrelu))
    pooled_second_conv_layer = MaxPool2DLayer(second_conv_layer, pool_size=(2,2), stride=2)
    conv_dis_output = ReshapeLayer(pooled_second_conv_layer, ([0], 11*64*64))


    text_input_dis = InputLayer(shape = (None, 11, 300), input_var = input_text)
    text_input_dis = ReshapeLayer(text_input_dis, ([0], 11*300))

    input_fc_dis = ConcatLayer([conv_dis_output, text_input_dis], axis=1)
    frst_hidden_layer = batch_norm(DenseLayer(input_fc_dis, 5000, nonlinearity=lrelu))
    frst_hidden_layer = DropoutLayer(frst_hidden_layer, p=0.35)
    second_hidden_layer = batch_norm(DenseLayer(frst_hidden_layer, 2500, nonlinearity=lrelu))
    second_hidden_layer = DropoutLayer(second_hidden_layer, p=0.35)
    final_output_dis = DenseLayer(second_hidden_layer, 2, nonlinearity = sigmoid)

    return final_output_dis


def gen_model(input_noise, input_text):
    # Generator model
    lrelu = LeakyRectify(0.1)

    input_gen_noise = InputLayer(shape=(None, 200), input_var=input_noise)
    input_gen_text = InputLayer(shape=(None, 11, 300), input_var=input_text)
    input_gen_text = ReshapeLayer(input_gen_text, ([0], 11*300))

    input_gen = ConcatLayer([input_gen_noise,  input_gen_text], axis=1)
    
    first_hidden_layer = batch_norm(DenseLayer(input_gen, 2500, nonlinearity=lrelu))
    first_hidden_layer = DropoutLayer(first_hidden_layer, p=0.15)

    second_hidden_layer = batch_norm(DenseLayer(first_hidden_layer, 5000, nonlinearity=lrelu))
    second_hidden_layer = DropoutLayer(second_hidden_layer, p=0.15)

    third_hidden_layer = DenseLayer(second_hidden_layer, 11*128*128, nonlinearity=lrelu)
    third_hidden_layer = ReshapeLayer(third_hidden_layer, ([0], 11, 128, 128))

    first_conv_layer = batch_norm(Deconv2DLayer(third_hidden_layer, 17, 5, stride=1, crop=2, nonlinearity=lrelu))
    second_conv_layer = Deconv2DLayer(first_conv_layer, 3, 5, stride=1, crop=2, nonlinearity=tanh)
    return second_conv_layer

'''
def scaleupArray(arr):
    w, h, c = arr.shape
    maxVal = np.max(arr, axis=(0,1))*1.0
    minVal = np.min(arr, axis=(0,1))*1.0
    for id in range(c):
        currMinVal = minVal[id]
        currMaxVal = maxVal[id]
        for l1 in range(w):
            for l2 in range(h):
                arr[l1][l2][id] = ((arr[l1][l2][id] - currMinVal)/(currMaxVal-currMinVal))*255
    return arr
'''
def scaleupArray(arr):

    arr = (arr+1.0/2.0)*255
    return arr

def train_network(num_epochs, batch_size, num_iters_inner):
    print "Start Training!"    

    input_noise = T.dmatrix('n')
    input_image = T.dtensor4('i')
    input_text = T.dtensor3('t')

    gen = gen_model(input_noise, input_text)
    disc = disc_model(input_image, input_text)

    real_img_val = lasagne.layers.get_output(disc)

    all_layers = lasagne.layers.get_all_layers(disc)
    #print all_layers
    fake_img_val = lasagne.layers.get_output(disc, {all_layers[0]: lasagne.layers.get_output(gen), all_layers[9]: input_text})

    gen_loss = lasagne.objectives.binary_crossentropy(fake_img_val, 1).mean()
    disc_loss = (lasagne.objectives.binary_crossentropy(real_img_val, 1)
            + lasagne.objectives.binary_crossentropy(fake_img_val, 0)).mean()
    
    gen_params = lasagne.layers.get_all_params(gen, trainable=True)
    disc_params = lasagne.layers.get_all_params(disc, trainable=True)
    
    update_gen = lasagne.updates.adam(gen_loss, gen_params, learning_rate=2.5e-4)
    update_disc = lasagne.updates.adam(disc_loss, disc_params, learning_rate=2.5e-4)

    train_disc_fn = theano.function([input_image, input_noise, input_text],
                               [(real_img_val >= .5).mean(),
                                (fake_img_val < .5).mean()],
                               updates=update_disc)

    train_gen_fn = theano.function([input_noise, input_text],
                               [(fake_img_val >= .5).mean()],
                               updates=update_gen)


    test_disc_fn = theano.function([input_image, input_noise, input_text],
                               [(lasagne.layers.get_output(disc, deterministic=True) >= .5).mean(),
                                (lasagne.layers.get_output(disc, {all_layers[0] : lasagne.layers.get_output(gen, deterministic=True), all_layers[9] : input_text}, deterministic=True) < .5).mean()])
    test_gen_fn = theano.function([input_noise, input_text],
                               [(lasagne.layers.get_output(disc, {all_layers[0] : lasagne.layers.get_output(gen, deterministic=True), all_layers[9] : input_text}, deterministic=True) >= .5).mean()])
    
    test_gen_fn_samples = theano.function([input_noise, input_text],
                                lasagne.layers.get_output(gen, deterministic=True))
    '''
    num_epochs = 24
    batch_size = 125
    iter_per_epoch = 1+X_train_img.shape[0]/batch_size
    num_iters_inner = 3
    '''
    iter_per_epoch = 1+X_train_img.shape[0]/batch_size
    count = 0
    
    print "Set-up system! Starting epochs!"
    for epoch in range(num_epochs):
        train_disc_acc = 0.0
        train_gen_acc = 0.0
        for itern in range(iter_per_epoch):
            for inner_itern in range(num_iters_inner):
                imgs, caption = get_sampled_batch_for_training(X_train_img, X_train_caption, batch_size)
                noise = np.random.rand(batch_size, 200)
                train_disc_acc += np.array(train_disc_fn(imgs, noise, caption))
            imgs, caption = get_sampled_batch_for_training(X_train_img, X_train_caption, batch_size)
            noise = np.random.rand(batch_size, 200)
            train_gen_acc += np.array(train_gen_fn(noise, caption))
            if count%10==0:
                print "Iters done : (", count, "/", (iter_per_epoch*num_epochs), ")"
            count += 1
        train_disc_acc /= (1.0 * num_iters_inner * iter_per_epoch)
        train_gen_acc /= (1.0 * iter_per_epoch) 
        print "Epoch done : (", (epoch+1), "/", num_epochs, ")"
        print "Train_disc_acc_avg = ", train_disc_acc
        print "Train_gen_acc_avg = ", train_gen_acc
        
        curr_noise = np.random.rand(X_train_img.shape[0], 200)
        print "Current_disc_acc = ", test_disc_fn(X_train_img, curr_noise, X_train_caption)
        print "Current_gen_acc = ", test_gen_fn(curr_noise, X_train_caption)

        if epoch==num_epochs-1:
            img_dc = {}
            curr_noise = np.random.rand(X_test_img.shape[0], 200)
            test_samples = np.array(test_gen_fn_samples(curr_noise, X_test_caption))
            print test_samples.shape
            for x in range(test_samples.shape[0]):
                img_dc[imageIdToNameDict[X_train_img.shape[0]+X_val_img.shape[0]+x]] = np.array(test_samples[x])
                arr = test_samples[x]
                c, w, h = arr.shape
                arr = np.reshape(arr, (w, h, c))
                arr_copy = np.array(arr)
                arr = scaleupArray(arr)
                arr = np.asarray(arr)
                arr_copy = np.asarray(arr_copy)
                im = Image.fromarray(np.uint8(arr))
                im.save("/home/kruti/text2image/data/answers_2_1/"+imageIdToNameDict[X_train_img.shape[0]+X_val_img.shape[0]+x])
                im2 = Image.fromarray(np.uint8(arr_copy))
                im2.save("/home/kruti/text2image/data/answers_2_2/"+imageIdToNameDict[X_train_img.shape[0]+X_val_img.shape[0]+x])
            np.save('test_images_pixel_values.npy', img_dc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', metavar='path', required=False, default=24)
    parser.add_argument('--batch_size', metavar='path', required=False, default=125)
    parser.add_argument('--num_iters_inner', metavar='path', required=False, default=3)
    args = parser.parse_args()
    
    X_train_img, X_train_caption, X_val_img, X_val_caption, X_test_img, X_test_caption = load_dataset()
    train_network(num_epochs=args.num_epochs, batch_size=args.batch_size, num_iters_inner=args.num_iters_inner)

