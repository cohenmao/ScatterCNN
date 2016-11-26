import numpy
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn

input_scales = T.vector("input_scales")
filter_scales = T.vector("filter_scales")

flt = T.ftensor4("flt")
res = T.ftensor4("res")

conved = dnn.dnn_conv(img=res, kerns=flt, subsample=(3, 3), border_mode=15)
pooled = dnn.dnn_pool(img=conved, ws=(4, 4), stride=(3, 3))

f = theano.function(inputs=[res, flt], outputs=pooled, allow_input_downcast=True)
print(numpy.shape(f(numpy.ones((1, 30, 113, 113)), numpy.ones((30, 30, 31, 31)))))

