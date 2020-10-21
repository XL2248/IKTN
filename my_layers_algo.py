import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.convolutional import Conv1D
from keras.layers import Concatenate
import numpy as np
import code
#from keras_pos_embd import TrigPosEmbedding

class Length(Layer):
    """
    Compute the lengths of capsules. 
    The values could be viewed as probability.
    """
    def call(self, inputs, **kwargs):
        probs = K.sqrt(K.sum(K.square(inputs), axis=-1, keepdims=True))
        #code.interact(local=locals())
        return probs
    def compute_mask(self, x, mask):
        return mask
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)

def squash1(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

def squash(s, axis=-1):
    """
    Squash function. This could be viewed as one kind of activations.
    """
    squared_s = K.sum(K.square(s), axis=axis, keepdims=True)
    scale = squared_s / (1 + squared_s) / K.sqrt(squared_s + K.epsilon())
    return scale * s

def softmax(x, axis=-1):
    """
    Self-defined softmax function
    """
    x = K.exp(x - K.max(x, axis=axis, keepdims=True))
    x /= K.sum(x, axis=axis, keepdims=True)
    return x

def position_embedding(inputs):
    batch_size = K.shape(inputs)[0]
    seq_len, output_dim = K.shape(inputs)[1], K.shape(inputs)[2]
#    code.interact(local=locals())
    pos_input = K.tile(K.expand_dims(K.arange(0, seq_len), axis=0), [batch_size, 1])
    pos_input = K.cast(pos_input, K.floatx())
    evens = K.arange(0, output_dim // 2) * 2
    odds = K.arange(0, output_dim // 2) * 2 + 1
    even_embd = K.sin(
       K.dot(
            K.expand_dims(pos_input, -1),
            K.expand_dims(1.0 / K.pow(
                10000.0,
               K.cast(evens, K.floatx()) / K.cast(output_dim, K.floatx())
           ), 0)
       )
    )
    odd_embd = K.cos(
        K.dot(
            K.expand_dims(pos_input, -1),
            K.expand_dims(1.0 / K.pow(
                10000.0, K.cast((odds - 1), K.floatx()) / K.cast(output_dim, K.floatx())
           ), 0)
       )
    )
#    code.interact(local=locals())
    embd = K.stack([even_embd, odd_embd], axis=-1)
    output = K.reshape(embd, [-1, K.shape(inputs)[1], output_dim])
    #code.interact(local=locals())
    #output = K.concatenate([inputs, output], axis=-1)
    output += inputs

    return output

class Capsule(Layer):
    def __init__(self, num_capsule, A, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        eye = A * K.zeros(self.num_capsule, dtype='float32') + K.eye(self.num_capsule, dtype='float32')
        shifted = tf.manip.roll(eye, shift=1, axis=0)
        self.A = A + eye + shifted
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)
    def compute_mask(self, x, mask):
        return mask

    def build(self, input_shape):
        super(Capsule, self).build(input_shape[1])
        self.input_dim_capsule = input_shape[1][-1]
        self.steps = input_shape[1][-2]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, self.input_dim_capsule,
                                            self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[1][-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1,
                                            self.input_dim_capsule,
                                            self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, vecs, mask=None):
        x1 = vecs[0]
        u_vecs = vecs[1] # B * L * Dim
        seq_len = self.steps
        input_dim_capsule = self.input_dim_capsule
        mask = mask[0]
        if mask is not None:
            mask  = K.expand_dims(mask, axis=-2)
            mask = K.cast(mask, K.floatx())
        if self.share_weights:
            temp = K.repeat_elements(self.W, self.steps, axis=0)
            tmp = K.permute_dimensions(temp, (1, 0, 2))
            in_posi = position_embedding(tmp)
            reshape_in_posi = K.reshape(in_posi, (input_dim_capsule, seq_len * self.dim_capsule))
            out_posi = K.expand_dims(reshape_in_posi, axis=0)
            u_hat_vecs = K.conv1d(u_vecs, out_posi)
        else:
            temp = K.repeat_elements(self.W, self.steps, axis=0) # seq_len * 600 * 300
            tmp = K.permute_dimensions(temp, (1, 0, 2)) # 600 * seq_len * 300
            in_posi = position_embedding(tmp) # 600 * seq_len * 300
            reshape_in_posi = K.reshape(in_posi, (input_dim_capsule, seq_len * self.dim_capsule)) # 600 * (seq_len*300)
            reshape_in_posi_ = K.expand_dims(reshape_in_posi, axis=0) # 1 * 600 * (seq_len*300)
            temp_posi = K.repeat_elements(reshape_in_posi_, self.steps, axis=0) # seq_len * 600 * (sel_len * 300)
            tmp_posi = K.permute_dimensions(temp_posi, (2, 0, 1)) #(seq_len*300) * seq_len * 600
            out_posi = position_embedding(tmp_posi) # (seq_len*300) * seq_len * 600
            in_out_posi = K.permute_dimensions(out_posi, (1, 2, 0))
            u_hat_vecs = K.local_conv1d(u_vecs, in_out_posi, [1], [1])
#        batch_size
        u_hat_vecs = position_embedding(u_hat_vecs)
        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            c = c * mask
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)