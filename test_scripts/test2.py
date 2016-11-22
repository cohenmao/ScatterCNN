import numpy
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn


image = T.ftensor4("image")
image2 = T.ftensor4("image2")
filter = T.ftensor4("filter")

# conved = dnn.dnn_conv(image, filter)
conved = dnn.dnn_conv(image, filter, subsample=(1, 1), border_mode=15)
f = theano.function(inputs=[image, filter], outputs=[conved], allow_input_downcast=True)
concatenated = T.concatenate((image, image2), axis=0)
g = theano.function(inputs=[image, image2], outputs=[concatenated], allow_input_downcast=True)
print(f(numpy.ones((128, 90, 227, 227)), numpy.ones((1080, 90, 32, 32))))
#print(numpy.shape(g(numpy.ones((2, 2, 2, 2)), numpy.zeros((3, 2, 2, 2)))))