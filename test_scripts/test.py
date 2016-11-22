import numpy
import theano
import theano.tensor as T

input_scales = T.vector("input_scales")
filter_scales = T.vector("filter_scales")



def greater_than_elementwise(X, y):

    res, _ = theano.scan(fn=lambda X, y: X > y, sequences=X, non_sequences=y, n_steps=X.shape[0])
    return res

is_apply_filter, _ = theano.scan(fn=lambda Y, X: greater_than_elementwise(X, Y),
                                 sequences=input_scales,
                                 non_sequences=filter_scales,
                                 n_steps=input_scales.shape[0]
                                 )


check = theano.function(inputs=[input_scales, filter_scales], outputs=T.flatten(is_apply_filter))
apply = check([1, 2, 3, 4], [3, 1, 5])
1


# TODO: try and define in advance the size of the filter_bank, the same way alexnet does in the filter shape which is not a symbolic variable



