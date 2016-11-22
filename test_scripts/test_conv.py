import theano
import numpy
from theano import tensor as T
from theano.sandbox.cuda import dnn
rng = numpy.random.RandomState(23455)

# parameters:
input_dims = (256, 3, 227, 227)
filter_dims = [(96, 3, 13, 13), (256, 96, 5, 5)] # 256, 96, x, x
convstride = [4, 1]
padsize = [0, 2]
poolsize = [3, 3]
poolstride = [2, 2]
layer1_input = T.ftensor4()
layer2_input = T.ftensor4()
filter1 = T.ftensor4()
filter2 = T.ftensor4()
# define theano variables and functions:
C1 = dnn.dnn_conv(img=layer1_input, kerns=filter1, subsample=(convstride[0], convstride[0]), border_mode=padsize[0])
D1 = dnn.dnn_pool(C1, ws=(poolsize[0], poolsize[0]), stride=(poolstride[0], poolstride[0]))
conv_space1 = theano.function(inputs=[layer1_input, filter1], outputs=C1)
pool1 = theano.function(inputs=[C1], outputs=D1)
C2 = dnn.dnn_conv(img=layer2_input, kerns=filter2, subsample=(convstride[1], convstride[1]), border_mode=padsize[1])
D2 = dnn.dnn_pool(C2, ws=(poolsize[1], poolsize[1]), stride=(poolstride[1], poolstride[1]))
conv_space2 = theano.function(inputs=[layer2_input, filter2], outputs=C2)
pool2 = theano.function(inputs=[C2], outputs=D2)
# create data:
A = numpy.asarray(rng.normal(0, 1, input_dims), dtype=numpy.float32)
B1 = numpy.asarray(rng.normal(0, 1, filter_dims[0]), dtype=numpy.float32)
B2 = numpy.asarray(rng.normal(0, 1, filter_dims[1]), dtype=numpy.float32)
# apply layers:
first_layer_output = pool1(conv_space1(A, B1))
second_layer_output = pool2(conv_space2(first_layer_output, B2))
# concatenate:
# layer3_input = T.concatenate([T.flatten(second_layer_output, 2), T.flatten(first_layer_output, 2)], axis=2)
# print dimension of convolution:
print numpy.shape(second_layer_output)
print numpy.shape(first_layer_output)
# print numpy.shape(layer3_input)

"""
layer1_output_shape = (30, 113, 113, 256)
layer1_output = numpy.asarray(rng.normal(0, 1, layer1_output_shape), dtype=numpy.float32)
layer1_output_pooled = dnn.dnnpool(layer1_output, ws=(9, 9), stride=(3, 3))
layer2_output_shape = (360, 4, 4, 256)
layer2_output = numpy.asarray(rng.normal(0, 1, layer2_output_shape), dtype=numpy.float32)
layer3_input = T.concatenate([T.flatten(layer2_output, 2), T.flatten(layer1_output_pooled, 2)])
print numpy.shape(layer3_input)
"""

"""
dat = T.ftensor4()
flt = T.ftensor4()
conv = dnn.dnn_conv(img=dat, kerns=flt)
conv_func = theano.function(inputs=[dat, flt], outputs=conv)

A = numpy.asarray(rng.normal(0, 1, (3, 4, 5, 6)), dtype=numpy.float32)
B = numpy.asarray(rng.normal(0, 1, (3, 4, 1, 1)), dtype=numpy.float32)
C = conv_func(A, B)
1
"""
