import numpy
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
import time

input = T.tensor4("input")
flt = T.tensor3('flt')
W = T.ftensor4("W")
output = dnn.dnn_conv(img=input, kerns=W, subsample=(1, 1), border_mode=1)

def convolove_feature_with_filter(input, filter):

    I = T.reshape(input, (1, 1, input.shape[0], input.shape[1]))
    f = T.reshape(filter, (1, 1, filter.shape[0], filter.shape[1]))
    conv = dnn.dnn_conv(img=I, kerns=f, subsample=(1, 1), border_mode='half')
    return T.reshape(conv, (conv.shape[2], conv.shape[3]))

def process_single_feature(feature, flt):

    single_feature_conved, _ = theano.scan(fn=lambda f, I: convolove_feature_with_filter(I, f),
                                           sequences=[flt],
                                           non_sequences=feature,
                                           n_steps=flt.shape[0]
                                           )
    return single_feature_conved

def process_single_image(image, flt):

    single_image_conved, _ = theano.scan(fn=lambda I, f: process_single_feature(I, f),
                                         sequences=[image],
                                         non_sequences=flt,
                                         n_steps=image.shape[0]
                                         )
    return T.reshape(single_image_conved, (image.shape[0]*flt.shape[0], single_image_conved.shape[2], single_image_conved.shape[3]))


conved, _ = theano.scan(fn=lambda I, f: process_single_image(I, f),
                        sequences=input,  # input -> I
                        non_sequences=[flt],  # flt -> f, weights -> w
                        n_steps=input.shape[0]
                        )

my_conv = theano.function(inputs=[input, flt], outputs=conved, allow_input_downcast=True)
alex_conv = theano.function(inputs=[input, W], outputs=output, allow_input_downcast=True)

input = numpy.ones((1, 3, 227, 227))
flt = numpy.ones((30, 64, 64))
W = numpy.ones((96, 3, 11, 11))
print('starting test')
a = time.time()
conved = alex_conv(input, W)
b = time.time()
print('alex convolution took %f seconds' % (b-a))
a = time.time()
conved = my_conv(input, flt)
b = time.time()
print('my convolution took %f seconds' % (b-a))



