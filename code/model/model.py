import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt

from lasagne.layers import *
from lasagne.nonlinearities import *


imageIdToNameDict = {}
imageIdToCaptionVectorDict = {}
X_train_img, X_train_caption, X_val_img, X_val_caption, X_test_img, X_test_caption = None, None, None, None, None, None


def load_dataset():
    pass


def get_sampled_batch_for_training(X_train_img, X_train_caption, batch_size, shuffle=True):
    pass
    


def disc_model(input_img, input_text):
    # disc layer description
    lrelu = LeakyRectify(0.1)

    input_dis = InputLayer(shape = (None, 3, 128, 128), input_var = input_img)
    frst_conv_layer = batch_norm(Conv2DLayer(input_dis, 20, 3, stride=1, pad=1, nonlinearity=lrelu))
    second_conv_layer = batch_norm(Conv2DLayer(frst_conv_layer, 15, 3, stride=1, pad=1, nonlinearity=lrelu))
    pooled_second_conv_layer = MaxPool2DLayer(second_conv_layer, pool_size=(2,2), stride=2)
    conv_dis_output = ReshapeLayer(pooled_second_conv_layer, ([0], 15*64*64))


    text_input_dis = InputLayer(shape = (None, 11, 300), input_var = input_text)
    text_input_dis = ReshapeLayer(text_input_dis, ([0], 11*300))

    input_fc_dis = ConcatLayer([conv_dis_output, text_input_dis], axis=1)
    frst_hidden_layer = batch_norm(DenseLayer(input_fc_dis, 10000, nonlinearity=lrelu))
    frst_hidden_layer = DropoutLayer(frst_hidden_layer, p=0.35)
    second_hidden_layer = batch_norm(DenseLayer(frst_hidden_layer, 5000, nonlinearity=lrelu))
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
    
    first_hidden_layer = batch_norm(DenseLayer(input_gen, 5000, nonlinearity=lrelu))
    first_hidden_layer = DropoutLayer(first_hidden_layer, p=0.15)

    second_hidden_layer = batch_norm(DenseLayer(first_hidden_layer, 10000, nonlinearity=lrelu))
    second_hidden_layer = DropoutLayer(second_hidden_layer, p=0.15)

    third_hidden_layer = DenseLayer(second_hidden_layer, 15*128*128, nonlinearity=lrelu)
    third_hidden_layer = ReshapeLayer(third_hidden_layer, ([0], 15, 128, 128))

    first_conv_layer = batch_norm(Deconv2DLayer(third_hidden_layer, 20, 3, stride=1, pad=1, nonlinearity=lrelu))
    second_conv_layer = Deconv2DLayer(first_conv_layer, 3, 3, stride=1, pad=1, nonlinearity=lrelu)
    return second_conv_layer



def train_network():
    
    input_noise = T.dmatrix()
    input_image = T.dtensor4()
    input_text = T.dtensor3()

    gen = gen_model(input_noise, input_text)
    disc = disc_model(input_image, input_text)

    real_img_val = lasagne.layers.get_output(disc)
    fake_img_val = lasagne.layers.get_output(disc, {input_img : lasagne.layers.get_output(gen), input_text : input_text})

    gen_loss = lasagne.objectives.binary_crossentropy(fake_img_val, 1).mean()
    disc_loss = (lasagne.objectives.binary_crossentropy(real_img_val, 1)
            + lasagne.objectives.binary_crossentropy(fake_img_val, 0)).mean()
    
    gen_params = lasagne.layers.get_all_params(gen, trainable=True)
    disc_params = lasagne.layers.get_all_params(disc, trainable=True)
    
    update_gen = lasagne.updates.adam(gen_loss, gen_params, learning_rate=1e-4)
    update_disc = lasagne.updates.adam(disc_loss, disc_params, learning_rate=1e-4)

    train_disc_fn = theano.function([input_image, input_noise, input_text],
                               [(real_img_val > .5).mean(),
                                (fake_img_val < .5).mean()],
                               updates=update_disc)

    train_gen_fn = theano.function([input_noise, input_text],
                               [(fake_img_val > .5).mean()],
                               updates=update_gen)


    test_disc_fn = theano.function([input_image, input_noise, input_text],
                               [(lasagne.layers.get_output(disc, deterministic=True) > .5).mean(),
                                (lasagne.layers.get_output(disc, {input_img : lasagne.layers.get_output(gen, deterministic=True), input_text : input_text}, deterministic=True) < .5).mean()])
    test_gen_fn = theano.function([input_noise, input_text],
                               [(lasagne.layers.get_output(disc, {input_img : lasagne.layers.get_output(gen, deterministic=True), input_text : input_text}, deterministic=True) > .5).mean()])
    
    test_gen_fn_samples = theano.function([input_noise, input_text],
                                lasagne.layers.get_output(gen, deterministic=True))

    num_epochs = 7
    batch_size = 200
    iter_per_epoch = X_train_img.shape[0]/batch_size
    num_iters_inner = 3
    count = 0
    for epoch in range(num_epochs):
        train_disc_acc = 0.0
        train_gen_acc = 0.0
        for itern in iter_per_epoch:
            for inner_itern in num_iters_inner:
                imgs, caption = get_sampled_batch_for_training(X_train_img, X_train_caption, batch_size, shuffle=True)
                noise = np.random.rand(batch_size, 200)
                train_disc_acc += np.array(train_disc_fn(imgs, noise, caption))
            imgs, caption = get_sampled_batch_for_training(X_train_img, X_train_caption, batch_size, shuffle=True)
            noise = np.random.rand(batch_size, 200)
            train_gen_acc += np.array(train_gen_fn(noise, caption))
            count += 1
            if count%100==0:
                print "Iters done : (", count, "/", (iter_per_epoch*num_epochs), ")"
        train_disc_acc /= (1.0 * num_iters_inner * iter_per_epoch)
        train_gen_acc /= (1.0 * iter_per_epoch) 
        print "Epoch done : (", (epoch+1), "/", num_epochs, ")"
        print "Train_disc_acc_avg = ", train_disc_acc
        print "Train_gen_acc_avg = ", train_gen_acc
        curr_noise = np.random.rand(X_train_img.shape[0], 200)
        print "Current_disc_acc = ", test_disc_fn(X_train_img, curr_noise, X_train_caption)
        print "Current_gen_acc = ", test_gen_fn(curr_noise, X_train_caption)

        if epoch==num_epochs-1:
            curr_noise = np.random.rand(X_test_img.shape[0], 200)
            test_samples = test_gen_fn_samples(curr_noise, X_test_caption)
            #plt.imsave('mnist_samples.png',
            #           (samples.reshape(6, 7, 28, 28)
            #                   .transpose(0, 2, 1, 3)
            #                   .reshape(6*28, 7*28)),
            #           cmap='gray')
            # TODO : save all test images - with id


X_train_img, X_train_caption, X_val_img, X_val_caption, X_test_img, X_test_caption = load_dataset()
train_network()

