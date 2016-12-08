import numpy
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn

input = T.tensor4("input")
flt = T.tensor3('flt')
W = T.ftensor4("W")

def process_single_featuremap(image, flt):

    reshaped_image = T.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))  # (1, X, X, n_im)
    shuffled_image = reshaped_image.dimshuffle(3, 0, 1, 2)  # (n_im, 1, X, X)
    reshaped_filter = T.reshape(flt, (1, flt.shape[0], flt.shape[1], flt.shape[2]))  # (1, n_flt, Y, Y)
    shuffled_filter = reshaped_filter.dimshuffle(1, 0, 2, 3)  # (n_flt, 1, Y, Y)
    conv = dnn.dnn_conv(shuffled_image, shuffled_filter, subsample=(3, 3), border_mode=15)  # (n_im, n_flt, Z, Z)
    return conv.dimshuffle(1, 2, 3, 0)  # (n_flt, Z, Z, n_im)

conv, _ = theano.scan(fn=lambda I, f: process_single_featuremap(I, f),
                      sequences=input,
                      non_sequences=[flt],
                      n_steps=input.shape[0]
                      )
conv = T.reshape(conv, (conv.shape[0]*conv.shape[1], conv.shape[2], conv.shape[3], conv.shape[4])).dimshuffle(3, 0, 1, 2)
pooled = dnn.dnn_pool(conv, ws=(6, 6), stride=(6, 6))
pooled = pooled.dimshuffle(1, 2, 3, 0)
my_conv = theano.function(inputs=[input, flt], outputs=pooled, allow_input_downcast=True)

input = numpy.ones((1, 227, 227, 1))
flt = numpy.ones((30, 31, 31))
print(numpy.shape(my_conv(input, flt)))
