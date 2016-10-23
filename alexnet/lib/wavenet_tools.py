import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn


def progress_single_image(input, filter_bank, W, padsize, convstride, filter_flag):
    """ function receives a single image data <input> and convolves it with <filter_bank>. <input> and <filter_bank>
    are both 3-dimensional tensors but the convolutions are 2-dimensional: each filter is separately convolved with each
    feature map in <input>

    :param input: 3-d tensor with the data from the previous layer
    :param filter_bank: 3-d tensor with the filters applied in this layer
    :param W: 2-d tensor with the weights for each filter-to-feature-map combination
    :param padsize: padding size for convolution
    :param convstride: convolution stride
    :param filter_flag: marks which convolutions represecnt frequency decreasing paths and should therefore be kept
    :return: 3-d tensor with convoloved data as described above
    """

    progressed, _ = theano.scan(fn=lambda I, W, f, cs, ps: progress_single_featuremap(I, f, W, cs, ps),
                                sequences=[input, W],
                                non_sequences=[filter_bank, W, convstride, padsize],
                                n_steps=input.shape[0]
                                )
    # reshape to a 3-d tensor with leading dimension corresponding to feature-filter combination - all filters
    # applied on the first feature, then all features on the second and so on:
    reshaped_data = T.reshape(progressed, (input.shape[0]*filter_bank.shape[0], input.shape[1], input.shape[2]))
    # return only frequency decreasing paths:
    return reshaped_data[T.nonzero(filter_flag)]


def progress_single_featuremap(input, filter_bank, W, padsize, convstride):
    """ function receives a single geature map of a single image <input> and convolves it with <filter_bank>.
    <input> is a 2-d tensor and <filter_vank> is a 3-dimensional tensor but the convolutions are 2-dimensional:
    each filter is separately convolved with the feature map in <input>

    :param input: 2-d tensor with the data from the previous layer
    :param filter_bank: 3-d tensor with the weighted filters applied in this layer
    :param W: 1-d tensor with the weights for filter
    :param padsize: padding size for convolution
    :param convstride: convolution stride
    :return: 3-d tensor with convoloved data as described above
    """

    # apply weights on the filter-to-response combinations:
    weighted_filter_bank, _ = theano.scan(fn=lambda X, y: X * y,
                                          sequences=[filter_bank, W],
                                          n_steps=filter_bank.shape[0]
                                          )
    progressed, _ = theano.scan(fn=lambda f, I, cs, ps: dnn.dnn_conv(img=I, kerns=f, subsample=cs, border_mode=ps),
                                sequences=weighted_filter_bank,
                                non_sequences=[input, convstride, padsize],
                                n_steps=filter_bank.shape[0]
                                )
    return progressed


def greater_than_elementwise(X, y):

    res, _ = theano.scan(fn=lambda X, y: X > y, sequences=X, non_sequences=y, n_steps=X.shape[0])
    return res
