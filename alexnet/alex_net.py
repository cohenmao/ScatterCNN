import sys
sys.path.append('./alexnet/lib')
import theano
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
theano.config.on_unused_input = 'warn'
import theano.tensor as T
from theano.sandbox.cuda import dnn

import numpy as np

from layers import DataLayer, ConvPoolLayer, FilterBankConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer


class AlexNet(object):

    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        flag_datalayer = config['use_data_layer']
        lib_conv = config['lib_conv']

        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        x = T.ftensor4('x')
        y = T.lvector('y')
        rand = T.fvector('rand')

        print '... building the model'
        self.layers = []
        params = []
        weight_types = []

        if flag_datalayer:
            data_layer = DataLayer(input=x, image_shape=(3, 256, 256,
                                                         batch_size),
                                   cropsize=227, rand=rand, mirror=True,
                                   flag_rand=config['rand_crop'])

            layer1_input = data_layer.output
        else:
            layer1_input = x

        convpool_layer1 = ConvPoolLayer(input=layer1_input,
                                        image_shape=(3, 227, 227, batch_size), 
                                        filter_shape=(3, 11, 11, 96), 
                                        convstride=4, padsize=0, group=1, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.0, lrn=True,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer1)
        params += convpool_layer1.params
        weight_types += convpool_layer1.weight_type

        convpool_layer2 = ConvPoolLayer(input=convpool_layer1.output,
                                        image_shape=(96, 27, 27, batch_size),
                                        filter_shape=(96, 5, 5, 256), 
                                        convstride=1, padsize=2, group=2, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.1, lrn=True,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer2)
        params += convpool_layer2.params
        weight_types += convpool_layer2.weight_type

        convpool_layer3 = ConvPoolLayer(input=convpool_layer2.output,
                                        image_shape=(256, 13, 13, batch_size),
                                        filter_shape=(256, 3, 3, 384), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=0, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer3)
        params += convpool_layer3.params
        weight_types += convpool_layer3.weight_type

        convpool_layer4 = ConvPoolLayer(input=convpool_layer3.output,
                                        image_shape=(384, 13, 13, batch_size),
                                        filter_shape=(384, 3, 3, 384), 
                                        convstride=1, padsize=1, group=2, 
                                        poolsize=1, poolstride=0, 
                                        bias_init=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer4)
        params += convpool_layer4.params
        weight_types += convpool_layer4.weight_type

        convpool_layer5 = ConvPoolLayer(input=convpool_layer4.output,
                                        image_shape=(384, 13, 13, batch_size),
                                        filter_shape=(384, 3, 3, 256), 
                                        convstride=1, padsize=1, group=2, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer5)
        params += convpool_layer5.params
        weight_types += convpool_layer5.weight_type

        fc_layer6_input = T.flatten(
            convpool_layer5.output.dimshuffle(3, 0, 1, 2), 2)
        fc_layer6 = FCLayer(input=fc_layer6_input, n_in=9216, n_out=4096)
        self.layers.append(fc_layer6)
        params += fc_layer6.params
        weight_types += fc_layer6.weight_type

        dropout_layer6 = DropoutLayer(fc_layer6.output, n_in=4096, n_out=4096)

        fc_layer7 = FCLayer(input=dropout_layer6.output, n_in=4096, n_out=4096)
        self.layers.append(fc_layer7)
        params += fc_layer7.params
        weight_types += fc_layer7.weight_type

        dropout_layer7 = DropoutLayer(fc_layer7.output, n_in=4096, n_out=4096)

        softmax_layer8 = SoftmaxLayer(
            input=dropout_layer7.output, n_in=4096, n_out=30)  # Maoz: 1000
        self.layers.append(softmax_layer8)
        params += softmax_layer8.params
        weight_types += softmax_layer8.weight_type

        # #################### NETWORK BUILT #######################

        self.cost = softmax_layer8.negative_log_likelihood(y)
        self.errors = softmax_layer8.errors(y)
        self.errors_top_5 = softmax_layer8.errors_top_x(y, 5)
        self.params = params
        self.x = x
        self.y = y
        self.rand = rand
        self.weight_types = weight_types
        self.batch_size = batch_size


class WaveNet(object):

    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        flag_datalayer = config['use_data_layer']
        lib_conv = config['lib_conv']
        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        x = T.ftensor4('x')
        y = T.lvector('y')
        filter_bank = T.ftensor3('filter_bank')
        filter_scale = T.fvector('filter_scale')
        input_scale = T.fvector('input_scale')
        #filter_bank = theano.shared(config['filter_bank'])
        #filter_scale = theano.shared(np.array(config['filter_scale']))
        #input_scale = theano.shared(np.zeros(3))
        rand = T.fvector('rand')

        print '... building the model'
        self.layers = []
        params = []
        weight_types = []

        if flag_datalayer:
            data_layer = DataLayer(input=x, image_shape=(3, 256, 256,
                                                         batch_size),
                                   cropsize=227, rand=rand, mirror=True,
                                   flag_rand=config['rand_crop'])

            layer1_input = data_layer.output
        else:
            layer1_input = x

        convpool_layer1 = FilterBankConvPoolLayer(input=layer1_input,
                                                  input_scales=input_scale,
                                                  image_shape=(3, 227, 227, batch_size),
                                                  filter_bank=filter_bank,
                                                  filter_scales=filter_scale,
                                                  filter_shape=(30, 31, 31),
                                                  convstride=1, padsize=15, group=1,
                                                  poolsize=3, poolstride=2,
                                                  bias_init=0.0, lrn=True,
                                                  lib_conv=lib_conv,
                                                  )   # layer output shape: (90, 113, 113, batch_size)
        self.layers.append(convpool_layer1)
        params += convpool_layer1.params
        weight_types += convpool_layer1.weight_type

        convpool_layer2 = FilterBankConvPoolLayer(input=convpool_layer1.output,
                                                  input_scales=convpool_layer1.output_scales,
                                                  image_shape=(90, 113, 113, batch_size),
                                                  filter_bank=filter_bank,
                                                  filter_scales=filter_scale,
                                                  filter_shape=(30, 31, 31),
                                                  convstride=9, padsize=15, group=1,
                                                  poolsize=4, poolstride=3,
                                                  bias_init=0.1, lrn=True,
                                                  lib_conv=lib_conv,
                                                  )  # layer output shape: (360, 4, 4, batch_size)
        self.layers.append(convpool_layer2)
        params += convpool_layer2.params
        weight_types += convpool_layer2.weight_type
        """
        fc_layer3_input = T.concatenate([T.flatten(convpool_layer2.output.dimshuffle(3, 0, 1, 2), 2),
                          T.flatten(
                              dnn.dnn_pool(convpool_layer1.output, ws=(9, 9), stride=(3, 3)).dimshuffle(3, 0, 1, 2), 2
                          )
                                         ]
                                        )
        """
        fc_layer3_input = T.flatten(
            convpool_layer2.output.dimshuffle(3, 0, 1, 2), 2)
        # 5760 = 4*4*360 (360-num feature maps in last layer, 4*4-response size)
        fc_layer3 = FCLayer(input=fc_layer3_input, n_in=4*4*360, n_out=2560)
        self.layers.append(fc_layer3)
        params += fc_layer3.params
        weight_types += fc_layer3.weight_type

        dropout_layer3 = DropoutLayer(fc_layer3.output, n_in=2560, n_out=2560)

        fc_layer4 = FCLayer(input=dropout_layer3.output, n_in=2560, n_out=2560)
        self.layers.append(fc_layer4)
        params += fc_layer4.params
        weight_types += fc_layer4.weight_type

        dropout_layer4 = DropoutLayer(fc_layer4.output, n_in=2560, n_out=2560)

        softmax_layer5 = SoftmaxLayer(
            input=dropout_layer4.output, n_in=2560, n_out=30)
        self.layers.append(softmax_layer5)
        params += softmax_layer5.params
        weight_types += softmax_layer5.weight_type

        # #################### NETWORK BUILT #######################

        self.cost = softmax_layer5.negative_log_likelihood(y)
        self.errors = softmax_layer5.errors(y)
        self.errors_top_5 = softmax_layer5.errors_top_x(y, 5)
        self.params = params
        self.x = x
        self.y = y
        self.filter_bank = filter_bank
        self.filter_scale = filter_scale
        self.input_scale = input_scale
        self.rand = rand
        self.weight_types = weight_types
        self.batch_size = batch_size


def compile_models(model, config, flag_top_5=False):

    x = model.x
    y = model.y
    filter_bank = model.filter_bank
    filter_scale = model.filter_scale
    input_scale = model.input_scale

    rand = model.rand
    weight_types = model.weight_types

    cost = model.cost
    params = model.params
    errors = model.errors
    errors_top_5 = model.errors_top_5
    batch_size = model.batch_size

    mu = config['momentum']
    eta = config['weight_decay']

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []

    learning_rate = theano.shared(np.float32(config['learning_rate']))
    lr = T.scalar('lr')  # symbolic learning rate

    if config['use_data_layer']:
        raw_size = 256
    else:
        raw_size = 227

    shared_x = theano.shared(np.zeros((3, raw_size, raw_size,
                                       batch_size),
                                      dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(np.zeros((batch_size,), dtype=int),
                             borrow=True)

    rand_arr = theano.shared(np.zeros(3, dtype=theano.config.floatX),
                             borrow=True)

    shared_filter_bank = theano.shared(config['filter_bank'], borrow=True)
    shared_filter_scale = theano.shared(config['filter_scale'], borrow=True)
    shared_input_scale = theano.shared(np.zeros(3, dtype=np.float32), borrow=True)

    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]

    if config['use_momentum']:

        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i, weight_type in \
                zip(params, grads, vels, weight_types):

            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")

            if config['use_nesterov_momentum']:
                vel_i_next = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad
            else:
                vel_i_next = mu * vel_i - real_lr * real_grad

            updates.append((vel_i, vel_i_next))
            updates.append((param_i, param_i + vel_i_next))

    else:
        for param_i, grad_i, weight_type in zip(params, grads, weight_types):
            if weight_type == 'W':
                updates.append((param_i,
                                param_i - lr * grad_i - eta * lr * param_i))
            elif weight_type == 'b':
                updates.append((param_i, param_i - 2 * lr * grad_i))
            else:
                raise TypeError("Weight Type Error")

    # Define Theano Functions

    train_model = theano.function([], cost, updates=updates,
                                  givens=[(x, shared_x), (y, shared_y),
                                          (filter_bank, shared_filter_bank),
                                          (filter_scale, shared_filter_scale),
                                          (input_scale, shared_input_scale),
                                          (lr, learning_rate),
                                          (rand, rand_arr)])

    validate_outputs = [cost, errors]
    if flag_top_5:
        validate_outputs.append(errors_top_5)

    validate_model = theano.function([], validate_outputs,
                                     givens=[(x, shared_x), (y, shared_y),
                                             (filter_bank, shared_filter_bank),
                                             (filter_scale, shared_filter_scale),
                                             (input_scale, shared_input_scale),
                                             (rand, rand_arr)])

    train_error = theano.function(
        [], errors, givens=[(x, shared_x), (y, shared_y),
                            (filter_bank, shared_filter_bank),
                            (filter_scale, shared_filter_scale),
                            (input_scale, shared_input_scale),
                            (rand, rand_arr)])

    return (train_model, validate_model, train_error,
            learning_rate, shared_x, shared_y, rand_arr, vels)
