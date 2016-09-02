# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:10:48 2016

@author: sicarbonnell

Code from: https://github.com/joelthchao/keras/blob/master/keras/layers/local.py
"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.layers import activations, initializations, regularizers, constraints, Reshape, BatchNormalization
from keras.layers.convolutional import conv_output_length
from keras.engine import Layer, InputSpec

def BatchNormalization_local(layer,tensor):
    shape = layer.output_shape
    return Reshape(shape[1:])(BatchNormalization(mode = 0,axis = 1)(Reshape((shape[1]*shape[2]*shape[3],))(tensor)))


class SemiShared(Layer):
    def __init__(self, nb_filter, shared_pool, nb_row=1, nb_col=1,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1),
                 dim_ordering=K.image_dim_ordering(),
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        if border_mode != 'valid':
            raise Exception('Invalid border mode for Convolution2D '
                            '(only "valid" is supported):', border_mode)
        if tuple(subsample) != (nb_row,nb_col): #model.to_json saves subsample as list and not as tuple
            raise Exception('Local layer only works with equal filter dimensions and strides')
        self.nb_filter = nb_filter
        self.shared_pool = shared_pool
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)

        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(SemiShared, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[2] % self.shared_pool[0] != 0 or input_shape[3] % self.shared_pool[1] != 0:
            raise Exception('Layer only works if input dimensions can be divided by shared pool dimensions')
        nb_x_pools = int(input_shape[2]/self.shared_pool[0])
        nb_y_pools = int(input_shape[3]/self.shared_pool[1])
        output_shape = self.get_output_shape_for(input_shape)
        if self.dim_ordering == 'th':
            _, nb_filter, output_row, output_col = output_shape
            input_filter = input_shape[1]
        elif self.dim_ordering == 'tf':
            _, output_row, output_col, nb_filter = output_shape
            input_filter = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        self.output_row = output_row
        self.output_col = output_col
        self.W_shape = (nb_filter,input_filter,nb_x_pools,nb_y_pools)#(output_row * output_col, self.nb_row * self.nb_col * input_filter, nb_filter)
        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))

        if self.bias:
            self.b = K.zeros((nb_filter,nb_x_pools, nb_y_pools), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        stride_row, stride_col = self.subsample
        nb_filter,_,_,_ = self.W_shape

        if self.dim_ordering == 'th':
            if K._backend == 'theano':
                x = x.reshape([x.shape[0],1,x.shape[1],x.shape[2],x.shape[3]])
                # x has shape (batchsize,1,input_nbfilter,input_rows,input_cols)
                
                W = K.repeat_elements(self.W, self.shared_pool[0], axis=2)
                W = K.repeat_elements(W, self.shared_pool[1], axis=3)
                # W has shape (nb_filter , input_nbfilter,input_rows,input_cols)
                
                output = K.sum(x*W,axis = 2) # uses broadcasting, sums over input filters

        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.bias:
            if self.dim_ordering == 'th':
                b = K.repeat_elements(self.b, self.shared_pool[0], axis=1)
                b = K.repeat_elements(b, self.shared_pool[1], axis=2)
                output += K.reshape(b, (1, nb_filter, self.output_row, self.output_col))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, self.output_row, self.output_col, nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'shared_pool': self.shared_pool,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(SemiShared, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
class LocallyConnected2D_fast(Layer):
    '''
        This version is changed from the original.
        It works only with subsample = filter size
        and if the input dimensions (rows,cols) are divisible by the filter size
        
        Notes: when using subsampling, pool(average) is used instead of pool(sum).
        theano sum mode doesn't work on my computer.
    '''
    
    '''LocallyConnected2D layer works almost the same as Convolution2D layer,
    except that weights are unshared, that is, a different set of filters is
    applied at each different patch of the input. When using this layer as the
    first layer in a model, provide the keyword argument `input_shape` (tuple
    of integers, does not include the sample axis), e.g.
    `input_shape=(3, 128, 128)` for 128x128 RGB pictures. Also, you will need
    to fix shape of the previous layer, since the weights can only be defined
    with determined output shape.

    # Examples
    ```python
        # apply a 3x3 unshared weights convolution with 64 output filters on a 32x32 image:
        model = Sequential()
        model.add(LocallyConnected2D(64, 3, 3, input_shape=(3, 32, 32)))
        # now model.output_shape == (None, 64, 30, 30)
        # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64 parameters

        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected2D(32, 3, 3))
        # now model.output_shape == (None, 32, 28, 28)
    ```

    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: Only support 'valid'. Please make good use of
            ZeroPadding2D to achieve same output shape.
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.
    '''
    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1),
                 dim_ordering=K.image_dim_ordering(),
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        if border_mode != 'valid':
            raise Exception('Invalid border mode for Convolution2D '
                            '(only "valid" is supported):', border_mode)
        if tuple(subsample) != (nb_row,nb_col): #model.to_json saves subsample as list and not as tuple
            raise Exception('Local layer only works with equal filter dimensions and strides')
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)

        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(LocallyConnected2D_fast, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[2] % self.nb_row != 0 or input_shape[3] % self.nb_col != 0:
            raise Exception('Layer only works if input dimensions can be divided by filter/stride dimensions')
        output_shape = self.get_output_shape_for(input_shape)
        if self.dim_ordering == 'th':
            _, nb_filter, output_row, output_col = output_shape
            input_filter = input_shape[1]
        elif self.dim_ordering == 'tf':
            _, output_row, output_col, nb_filter = output_shape
            input_filter = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        self.output_row = output_row
        self.output_col = output_col
        self.W_shape = (nb_filter,input_filter,input_shape[2],input_shape[3])#(output_row * output_col, self.nb_row * self.nb_col * input_filter, nb_filter)
        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))

        if self.bias:
            self.b = K.zeros((output_row, output_col, nb_filter), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        stride_row, stride_col = self.subsample
        nb_filter,_,_,_ = self.W_shape

        if self.dim_ordering == 'th':
            if K._backend == 'theano':
                x = x.reshape([x.shape[0],1,x.shape[1],x.shape[2],x.shape[3]])
                # x has shape (batchsize,1,input_nbfilter,input_rows,input_cols)
                # W has shape (nb_filter , input_nbfilter,input_rows,input_cols)
                output = K.sum(x*self.W,axis = 2) # uses broadcasting, sums over input filters
                if stride_row>1 or stride_col >1:
                    # sum pooling isn't working -> avg pooling multiplied by number of elements/pool
                    output = (stride_row*stride_col)*K.pool2d(output,(stride_row, stride_col),(stride_row, stride_col),pool_mode = 'avg')
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, nb_filter, self.output_row, self.output_col))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, self.output_row, self.output_col, nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(LocallyConnected2D_fast, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))