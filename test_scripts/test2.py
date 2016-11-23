import numpy
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn

# reshaping a tensor:
"""
x = T.ftensor3("x")
y = T.tile(x, (3, 1, 1))
z = T.reshape(T.flatten(T.reshape(y, (3, x.shape[0], x.shape[1], x.shape[2])).dimshuffle(1, 0, 2, 3)), (3, x.shape[0], x.shape[1], x.shape[2]))
w = z.shape
f = theano.function(inputs=[x], outputs=[y], allow_input_downcast=True)
g = theano.function(inputs=[x], outputs=[z], allow_input_downcast=True)
h = theano.function(inputs=[x], outputs=[w], allow_input_downcast=True)
# print(f(numpy.array([[[1, 2], [3, 4]], [[1, 3], [3, 5]]])))
print(g(numpy.array([[[1, 2], [3, 4]], [[-2, 10], [9, 5]], [[3, 7], [-1, 5]]])))
print(h(numpy.array([[[1, 2], [3, 4]], [[-2, 10], [9, 5]], [[3, 7], [-1, 5]]])))
"""

# transforming a matrix of weights to a tensor of weights:
# this assumes that matrix_ij is the weight of the i'th filter on the j'th response
weights = T.fmatrix("weights")
reshaped_weights = T.reshape(weights, (weights.shape[0], weights.shape[1], 1, 1))
y = T.tile(reshaped_weights, (1, 1, 3, 3))
z = T.reshape(y, (weights.shape[0], weights.shape[1], 3, 3))
f = theano.function(inputs=[weights], outputs=z, allow_input_downcast=True)
g = f(numpy.array([[2, 3, 5], [4, 6, 7]]))
print(g)

"""
# multiplying along an axis:
A = T.ftensor3("A")
B = T.ftensor3("B")
f = theano.function(inputs=[A, B], outputs=[A*B], allow_input_downcast=True)
print(f(numpy.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]), numpy.array([[[2, 2], [2, 2]], [[3, 3], [3, 3]]])))
"""