import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn


def progress_single_image(input, filter_bank, weights, biasvals, convstride, padsize, filter_flag):
    """ function receives a single image data <input> and convolves it with <filter_bank>. <input> and <filter_bank>
    are both 3-dimensional tensors but the convolutions are 2-dimensional: each filter is separately convolved with each
    feature map in <input>

    :param input: 3-d tensor with the data from the previous layer
    :param filter_bank: 3-d tensor with the filters applied in this layer
    :param weights: 2-d tensor with the weights for each filter-to-feature-map combination
    :param biasvals: 2-d tensor with the bias values for each filter-to-feature-map combination
    :param padsize: padding size for convolution
    :param convstride: convolution stride
    :param filter_flag: marks which convolutions represent frequency decreasing paths and should therefore be kept
    :return: 3-d tensor with convoloved data as described above
    """

    progressed, _ = theano.scan(fn=lambda I, W, b, f, cs, ps: progress_single_featuremap(I, f, W, b, cs, ps),
                                sequences=[input, weights, biasvals],
                                non_sequences=[filter_bank, convstride, padsize],
                                n_steps=input.shape[0]
                                )
    # reshape to a 3-d tensor with leading dimension corresponding to feature-filter combination - all filters
    # applied on the first feature, then all features on the second and so on:
    final_progressed = progressed[-1]
    reshaped_data = T.reshape(final_progressed, (input.shape[0]*filter_bank.shape[0], input.shape[1], input.shape[2]))
    # return only frequency decreasing paths:
    return reshaped_data[T.nonzero(filter_flag)]


def progress_single_featuremap(input, filter_bank, weights, biasvals, convstride, padsize):
    """ function receives a single geature map of a single image <input> and convolves it with <filter_bank>.
    <input> is a 2-d tensor and <filter_vank> is a 3-dimensional tensor but the convolutions are 2-dimensional:
    each filter is separately convolved with the feature map in <input>

    :param input: 2-d tensor with the data from the previous layer
    :param filter_bank: 3-d tensor with the weighted filters applied in this layer
    :param weights: 1-d tensor with the weights for filter-feature combinations
    :param biasvals: 1-d tensor with the bias values for filter-feature combinations
    :param padsize: padding size for convolution
    :param convstride: convolution stride
    :return: 3-d tensor with convoloved data as described above
    """

    # apply weights on the filter-to-response combinations:
    weighted_filter_bank, _ = theano.scan(fn=lambda X, y: X * y,
                                          sequences=[filter_bank, weights],
                                          n_steps=filter_bank.shape[0]
                                          )

    progressed, _ = theano.scan(fn=lambda f, b, I, cs, ps: convolove_feature_with_filter(I, f, b, cs, ps),
                                sequences=[weighted_filter_bank, biasvals],
                                non_sequences=[input, convstride, padsize],
                                n_steps=filter_bank.shape[0]
                                )
    return progressed[-1]


def convolove_feature_with_filter(input, filter, bias, convstride, padsize):
    """ performs convolution between 2-d feature map and 2-d weighted filter and adds the bias value

    :param input: 2-d tensor with the data from the previous layer
    :param filter: 2-d tensor with the weighted filters applied in this layer
    :param bias: 1-d tensor with the bias values for filter-feature combinations
    :param padsize: padding size for convolution
    :param convstride: convolution stride
    :return: 2-d tensor with convoloved data as described above
    """

    pad = int(padsize.get_value())
    stride = (int(convstride.get_value()), int(convstride.get_value()))
    I = T.reshape(input, (1, 1, input.shape[0], input.shape[1]))
    f = T.reshape(filter, (1, 1, filter.shape[0], filter.shape[1]))
    conv = dnn.dnn_conv(img=I, kerns=f, subsample=stride, border_mode=pad) + bias
    return T.reshape(conv, (conv.shape[2], conv.shape[3]))


def greater_than_elementwise(X, y):

    res, _ = theano.scan(fn=lambda X, y: X > y, sequences=X, non_sequences=y, n_steps=X.shape[0])
    return res[-1]


def process_single_featuremap(image, flt, convstride, padsize):

    reshaped_image = T.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))  # (1, X, X, n_im)
    shuffled_image = reshaped_image.dimshuffle(3, 0, 1, 2)  # (n_im, 1, X, X)
    reshaped_filter = T.reshape(flt, (1, flt.shape[0], flt.shape[1], flt.shape[2]))  # (1, n_flt, Y, Y)
    shuffled_filter = reshaped_filter.dimshuffle(1, 0, 2, 3)  # (n_flt, 1, Y, Y)

    pad = int(padsize.get_value())
    stride = (int(convstride.get_value()), int(convstride.get_value()))
    conv = dnn.dnn_conv(shuffled_image, shuffled_filter, subsample=stride, border_mode=pad)

    return conv.dimshuffle(1, 2, 3, 0)  # (n_flt, Z, Z, n_im)