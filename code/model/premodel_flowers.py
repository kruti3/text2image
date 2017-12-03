
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from data_utils import *


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.


imageIdToNameDict = {}
imageIdToCaptionVectorDict = {}
X_train, X_train_caption, X_val, X_val_caption, X_test, X_test_caption = None, None, None, None, None, None


pixelSz = 32


def load_dataset(tanh_flag):
    train_img_raw, val_img_raw, test_img_raw = imgtobin_flowers(tanh_flag)

    #print test_img_raw
    
    train_caption = np.load('/home/kruti/text2image/data/flowers/system_input_train.npy').item()
    validate_caption = np.load('/home/kruti/text2image/data/flowers/system_input_validate.npy').item()
    test_caption = np.load('/home/kruti/text2image/data/flowers/system_input_test.npy').item()

    train_sz = len(train_caption)
    X_train_caption_lcl = np.zeros((train_sz, 1 , 300))
    X_train_img_lcl = np.zeros((train_sz, 3 , pixelSz, pixelSz))
    validate_sz = len(validate_caption)
    X_val_caption_lcl = np.zeros((validate_sz, 1 , 300))
    X_val_img_lcl = np.zeros((validate_sz, 3 , pixelSz, pixelSz))
    test_sz = len(test_caption)
    X_test_caption_lcl = np.zeros((test_sz, 1 , 300))
    X_test_img_lcl = np.zeros((test_sz, 3 , pixelSz, pixelSz))
    
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


# ##################### Build the neural network model #######################
# We create two models: The generator and the discriminator network. The
# generator needs a transposed convolution layer defined first.

class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
            nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.Orthogonal(),
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)

def build_generator(input_noise, input_text, layer_list, fclayer_list):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm, ConcatLayer
    from lasagne.nonlinearities import sigmoid, LeakyRectify
    
    lrelu = LeakyRectify(0.1)

    input_gen_noise = InputLayer(shape=(None, 50), input_var=input_noise)
    input_gen_text = InputLayer(shape=(None, 1, 300), input_var=input_text)
    input_gen_text = ReshapeLayer(input_gen_text, ([0], 1*300))

    input_gen = ConcatLayer([input_gen_noise,  input_gen_text], axis=1)
    
    zeroth_hidden_layer = batch_norm(DenseLayer(input_gen, fclayer_list[0] , nonlinearity=lrelu))
   
    first_hidden_layer = batch_norm(DenseLayer(zeroth_hidden_layer, fclayer_list[1], nonlinearity=lrelu))
   
    second_hidden_layer = batch_norm(DenseLayer(first_hidden_layer, fclayer_list[2], nonlinearity=lrelu))
   
    third_hidden_layer = DenseLayer(second_hidden_layer, layer_list[0]*pixelSz*pixelSz, nonlinearity=lrelu)
    third_hidden_layer = ReshapeLayer(third_hidden_layer, ([0], layer_list[0], pixelSz, pixelSz))

    first_conv_layer = batch_norm(Deconv2DLayer(third_hidden_layer, layer_list[1], 5, stride=1, pad=2, nonlinearity=lrelu))
    second_conv_layer = batch_norm(Deconv2DLayer(first_conv_layer, layer_list[2], 5, stride=1, pad=2, nonlinearity=lrelu))
    #third_conv_layer = batch_norm(Deconv2DLayer(second_conv_layer, layer_list[3], 5, stride=1, crop=2, nonlinearity=lrelu))
    #fourth_conv_layer = batch_norm(Deconv2DLayer(third_conv_layer, layer_list[4], 5, stride=1, crop=2, nonlinearity=lrelu))
    fifth_conv_layer = Deconv2DLayer(second_conv_layer, 3, 5, stride=1, pad=2, nonlinearity=sigmoid)
    
    return fifth_conv_layer

def build_discriminator(input_img, input_text, layer_list, fclayer_list):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer, batch_norm, ConcatLayer)
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    lrelu = LeakyRectify(0.2)
    # input: (None, 3, 28, 28)

    input_dis = InputLayer(shape = (None, 3, pixelSz, pixelSz), input_var = input_img)
    #frst_conv_layer =  batch_norm(Conv2DLayer(input_dis, layer_list[4], 5, stride=1, pad=2, nonlinearity=lrelu))
    #second_conv_layer = batch_norm(Conv2DLayer(frst_conv_layer, layer_list[3], 5, stride=1, pad=2, nonlinearity=lrelu))
    third_conv_layer = batch_norm(Conv2DLayer(input_dis, layer_list[2], 5, stride=1, pad=2, nonlinearity=lrelu))
    fourth_conv_layer = batch_norm(Conv2DLayer(third_conv_layer, layer_list[1], 5, stride=1, pad=2, nonlinearity=lrelu))
    fifth_conv_layer = batch_norm(Conv2DLayer(fourth_conv_layer, layer_list[0], 5, stride=1, pad=2, nonlinearity=lrelu))
    #pooled_fifth_conv_layer = MaxPool2DLayer(fifth_conv_layer, pool_size=(2,2), stride=2)
    conv_dis_output = ReshapeLayer(fifth_conv_layer, ([0], layer_list[0]*pixelSz*pixelSz))

    text_input_dis = InputLayer(shape = (None, 1, 300), input_var = input_text)
    text_input_dis = ReshapeLayer(text_input_dis, ([0], 1*300))

    input_fc_dis = ConcatLayer([conv_dis_output, text_input_dis], axis=1)
    frst_hidden_layer = batch_norm(DenseLayer(input_fc_dis, fclayer_list[2], nonlinearity=lrelu))
    second_hidden_layer = batch_norm(DenseLayer(frst_hidden_layer, fclayer_list[1], nonlinearity=lrelu))
    third_hidden_layer = batch_norm(DenseLayer(second_hidden_layer, fclayer_list[0], nonlinearity=lrelu))
    final_output_dis = DenseLayer(third_hidden_layer, 1, nonlinearity = sigmoid)

    return final_output_dis

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, text, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], text[excerpt], 

def scaleTrain(arr):

    sz = 0
    for dirname, dirnames, filenames in os.walk('/home/kruti/text2image/data/flowers/flowerSamplesResized/train'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                    sz+=1
    dummy_arr = np.zeros((sz, pixelSz, pixelSz, 3))
    ct=0
    for dirname, dirnames, filenames in os.walk('/home/kruti/text2image/data/flowers/flowerSamplesResized/train'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                pix = Image.open(os.path.join(dirname, filename))
                pix = np.array(pix, dtype=np.float)
                dummy_arr[ct] = pix
                ct+=1

    mean = np.mean(dummy_arr, axis=0)
    std = np.std(dummy_arr, axis=0)

    arr = (arr*std)+mean
    arr[arr<0.0] = 0.0
    arr[arr>255.0] = 255.0
    return arr

def scaleActualRange(arr):
    w, h, c = arr.shape
    maxVal = np.max(arr, axis=(0,1))*1.0
    minVal = np.min(arr, axis=(0,1))*1.0
    for id in range(c):
        currMinVal = minVal[id]
        currMaxVal = maxVal[id]
        for l1 in range(w):
            for l2 in range(h):
                arr[l1][l2][id] = ((arr[l1][l2][id] - currMinVal)/(currMaxVal-currMinVal))*255.0
    arr[arr<0.0] = 0.0
    arr[arr>255.0] = 255.0
    return arr

def scaleActualRangeChanged(arr):
    w, h, c = arr.shape
    maxVal = np.max(arr, axis=(0,1))*1.0
    minVal = np.min(arr, axis=(0,1))*1.0
    for id in range(c):
        currMinVal = minVal[id]
        currMaxVal = maxVal[id]
        for l1 in range(w):
            for l2 in range(h):
                arr[l1][l2][id] = ((arr[l1][l2][id] * (currMaxVal-currMinVal)) + currMinVal) *255.0
    arr[arr<0.0] = 0.0
    arr[arr>255.0] = 255.0
    return arr

def scaleRange(arr):

    arr = (arr)*255
    return arr
# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(layer_list, fclayer_list, num_epochs, loss_func):
    # Load the dataset
    initial_eta=2e-4
    print("Loading data...")
    X_train, X_train_text, X_val, X_val_text, X_test, X_test_text = load_dataset(1)

    # Prepare Theano variables for inputs and targets
    noise_var = T.dmatrix('noise')
    input_img = T.dtensor4('inputs')
    input_text = T.dtensor3('text')
#    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(noise_var, input_text, layer_list, fclayer_list)
    discriminator = build_discriminator(input_img, input_text, layer_list, fclayer_list)

    all_layers = lasagne.layers.get_all_layers(discriminator)
    print ("LAYERS: ")
    print (all_layers)

    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator,
            {all_layers[0]: lasagne.layers.get_output(generator), all_layers[11]: input_text})

    # Create loss expressions
    generator_loss = None
    discriminator_loss = None

    if loss_func==0:
        generator_loss = lasagne.objectives.squared_error(fake_out, 1).mean()
        discriminator_loss = (lasagne.objectives.squared_error(real_out, 1) + lasagne.objectives.squared_error(fake_out, 0)).mean()
    else:
        generator_loss = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()
        discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 1) + lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()
    
    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    updates = lasagne.updates.adam(
            generator_loss, generator_params, learning_rate=eta, beta1=0.5)
    updates.update(lasagne.updates.adam(
            discriminator_loss, discriminator_params, learning_rate=eta, beta1=0.5))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([noise_var, input_img, input_text],
                               [(real_out > .5).mean(),
                                (fake_out < .5).mean()],
                               updates=updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var, input_text],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, X_train_text, 128, shuffle=True):
            inputs, text = batch
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 50))
            train_err += np.array(train_fn(noise, inputs, text))
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{}".format(train_err / train_batches))

        # And finally, we plot some generated data
        samples = np.array(gen_fn(lasagne.utils.floatX(np.random.rand(X_test_caption.shape[0], 50)), X_test_caption))
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            img_dc = {}
            offset = X_train.shape[0]+X_val.shape[0]
            for x in range(samples.shape[0]):
                img_dc[imageIdToNameDict[offset+x]] = np.array(samples[x])
                arr = samples[x]
                c, w, h = arr.shape
                arr = np.reshape(arr, (w, h, c))
                
                arr1 = np.asarray(scaleTrain(np.copy(arr)))
                im = Image.fromarray(np.uint8(arr1))
                im.save("/home/kruti/text2image/data/flowers/run3/1/"+str(epoch)+imageIdToNameDict[offset+x])
                
                arr2 = np.asarray(scaleRange(np.copy(arr)))
                im = Image.fromarray(np.uint8(arr2))
                im.save("/home/kruti/text2image/data/flowers/run3/2/"+str(epoch)+imageIdToNameDict[offset+x])
                
                arr3 = np.asarray(scaleActualRange(np.copy(arr)))
                im = Image.fromarray(np.uint8(arr3))
                im.save("/home/kruti/text2image/data/flowers/run3/3/"+str(epoch)+imageIdToNameDict[offset+x])
               
                arr4 = np.asarray(scaleActualRangeChanged(np.copy(arr)))
                im = Image.fromarray(np.uint8(arr4))
                im.save("/home/kruti/text2image/data/flowers/run3/4/"+str(epoch)+imageIdToNameDict[offset+x])
                

        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

    # Optionally, you could now dump the network weights to a file like this:
    #np.savez('mnist_gen.npz', *lasagne.layers.get_all_param_values(generator))
    #np.savez('mnist_disc.npz', *lasagne.layers.get_all_param_values(discriminator))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_list', required=True, nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument('--fclayer_list', required=True, nargs='+', type=int, default=[750, 1500, 2500])
    parser.add_argument('--num_epochs', required=False, type=int, default=1)
    parser.add_argument('--loss_func', required=False, type=int, default=1)
    args = parser.parse_args()
    
    main(layer_list=args.layer_list, fclayer_list=args.fclayer_list, num_epochs=args.num_epochs, loss_func=args.loss_func)
