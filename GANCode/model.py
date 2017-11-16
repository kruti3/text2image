import numpy as np
import theano
import theano.tensor as T
import lasagne

from lasagne.layers import *
from lasagne.nonlinearities import *




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


# Generator model
lrelu = LeakyRectify(0.1)

input_gen = InputLayer(shape=(None, 3500), input_var=input_text_var)

first_hidden_layer = batch_norm(DenseLayer(input_gen, 5000, nonlinearity=lrelu))
first_hidden_layer = DropoutLayer(first_hidden_layer, p=0.35)

second_hidden_layer = batch_norm(DenseLayer(first_hidden_layer, 10000, nonlinearity=lrelu))
second_hidden_layer = DropoutLayer(second_hidden_layer, p=0.35)

third_hidden_layer = DenseLayer(second_hidden_layer, 15*128*128, nonlinearity=lrelu)
third_hidden_layer = ReshapeLayer(third_hidden_layer, ([0], 15, 128, 128))

first_conv_layer = batch_norm(Deconv2DLayer(third_hidden_layer, 15, 3, stride=1, pad=1, nonlinearity=lrelu))
second_conv_layer = batch_norm(Deconv2DLayer(first_conv_layer, 20, 3, stride=1, pad=1, nonlinearity=lrelu))
