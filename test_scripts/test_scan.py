import theano
import theano.tensor as T
import numpy

a = T.ftensor4("a")
b = T.ftensor4("b")
c = T.flatten(a, 2)
d = T.flatten(b, 2)
e = T.concatenate([c, d], axis=1)
f = theano.function(inputs=[a, b], outputs=e, allow_input_downcast=True)
print(numpy.shape(f(numpy.ones((32, 30, 12, 12)), numpy.zeros((32, 15, 12, 12)))))