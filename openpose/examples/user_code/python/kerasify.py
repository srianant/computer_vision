import numpy as np
import struct

LAYER_DENSE = 1
LAYER_CONV2D = 2
LAYER_FLATTEN = 3
LAYER_ELU = 4
LAYER_ACTIVATION = 5
LAYER_MAXPOOLING2D = 6
LAYER_LSTM = 7
LAYER_EMBEDDING = 8
LAYER_TIMEDISTRIBUTED = 9

ACTIVATION_LINEAR = 1
ACTIVATION_RELU = 2
ACTIVATION_SOFTPLUS = 3
ACTIVATION_SIGMOID = 4
ACTIVATION_TANH = 5
ACTIVATION_HARD_SIGMOID = 6
ACTIVATION_SOFTMAX = 7

def write_floats(file, floats):
    '''
    Writes floats to file in 1024 chunks.. prevents memory explosion
    writing very large arrays to disk when calling struct.pack().
    '''
    step = 1024
    written = 0

    for i in np.arange(0, len(floats), step):
        remaining = min(len(floats) - i, step)
        written += remaining
        file.write(struct.pack('=%sf' % remaining, *floats[i:i+remaining]))

    assert written == len(floats)

def export_model(model, filename):
    with open(filename, 'wb') as f:

        def write_activation(activation):
            if activation == 'linear':
                f.write(struct.pack('I', ACTIVATION_LINEAR))
            elif activation == 'relu':
                f.write(struct.pack('I', ACTIVATION_RELU))
            elif activation == 'softplus':
                f.write(struct.pack('I', ACTIVATION_SOFTPLUS))
            elif activation == 'softmax':
                f.write(struct.pack('I', ACTIVATION_SOFTMAX))
            elif activation == 'tanh':
                f.write(struct.pack('I', ACTIVATION_TANH))
            elif activation == 'sigmoid':
                f.write(struct.pack('I', ACTIVATION_SIGMOID))
            elif activation == 'hard_sigmoid':
                f.write(struct.pack('I', ACTIVATION_HARD_SIGMOID))
            else:
                assert False, "Unsupported activation type: %s" % activation

        model_layers = [l for l in model.layers if type(l).__name__ not in ['Dropout']]
        num_layers = len(model_layers)
        f.write(struct.pack('I', num_layers))

        for layer in model_layers:
            layer_type = type(layer).__name__

            if layer_type == 'Dense':

                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]

                activation = layer.get_config()['activation']

                f.write(struct.pack('I', LAYER_DENSE))
                f.write(struct.pack('I', weights.shape[0]))
                f.write(struct.pack('I', weights.shape[1]))
                f.write(struct.pack('I', biases.shape[0]))

                weights = weights.flatten()
                biases = biases.flatten()

                write_floats(f, weights)
                write_floats(f, biases)

                write_activation(activation)

            elif layer_type == 'TimeDistributed':

                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]

                activation = layer.get_config()['layer']['config']['activation']

                f.write(struct.pack('I', LAYER_TIMEDISTRIBUTED))
                f.write(struct.pack('I', weights.shape[0]))
                f.write(struct.pack('I', weights.shape[1]))
                f.write(struct.pack('I', biases.shape[0]))

                weights = weights.flatten()
                biases = biases.flatten()

                write_floats(f, weights)
                write_floats(f, biases)

                write_activation(activation)

            elif layer_type == 'Conv2D':
                assert layer.border_mode == 'valid', "Only border_mode=valid is implemented"

                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                activation = layer.get_config()['activation']

                # The kernel is accessed in reverse order. To simplify the C side we'll
                # flip the weight matrix for each kernel.
                weights = weights[:,:,::-1,::-1]

                f.write(struct.pack('I', LAYER_CONV2D))
                f.write(struct.pack('I', weights.shape[0]))
                f.write(struct.pack('I', weights.shape[1]))
                f.write(struct.pack('I', weights.shape[2]))
                f.write(struct.pack('I', weights.shape[3]))
                f.write(struct.pack('I', biases.shape[0]))

                weights = weights.flatten()
                biases = biases.flatten()

                write_floats(f, weights)
                write_floats(f, biases)

                write_activation(activation)

            elif layer_type == 'Flatten':
                f.write(struct.pack('I', LAYER_FLATTEN))

            elif layer_type == 'ELU':
                f.write(struct.pack('I', LAYER_ELU))
                f.write(struct.pack('f', layer.alpha))

            elif layer_type == 'Activation':
                activation = layer.get_config()['activation']

                f.write(struct.pack('I', LAYER_ACTIVATION))
                write_activation(activation)

            elif layer_type == 'MaxPooling2D':
                assert layer.border_mode == 'valid', "Only border_mode=valid is implemented"

                pool_size = layer.get_config()['pool_size']

                f.write(struct.pack('I', LAYER_MAXPOOLING2D))
                f.write(struct.pack('I', pool_size[0]))
                f.write(struct.pack('I', pool_size[1]))

            elif layer_type == 'LSTM':
                inner_activation = layer.get_config()['recurrent_activation']
                activation = layer.get_config()['activation']
                return_sequences = int(layer.get_config()['return_sequences'])

                weights = layer.get_weights()

                input_dim = (weights[0].shape[1])/4 # say 40. Keep sample size multiples of 4

                # Kernel
                W_i = weights[0][:, 0:input_dim]  # [0:40] has input (i)
                W_f = weights[0][:, input_dim:(2 * input_dim)]  # [40:80] has f
                W_c = weights[0][:, (2 * input_dim):(3 * input_dim)]  # [80:120] has c
                W_o = weights[0][:, (3 * input_dim):]  # [80:120] has o

                # Recurrent Kernel
                U_i = weights[1][:, 0:input_dim]
                U_f = weights[1][:, input_dim:(2 * input_dim)]
                U_c = weights[1][:, (2 * input_dim):(3 * input_dim)]
                U_o = weights[1][:, (3 * input_dim):]

                # Use Bias = true, which is default 
                b_i = weights[2][0:input_dim]
                b_f = weights[2][input_dim:(2 * input_dim)]
                b_c = weights[2][(2 * input_dim):(3 * input_dim)]
                b_o = weights[2][(3 * input_dim):]

                f.write(struct.pack('I', LAYER_LSTM))
                f.write(struct.pack('I', W_i.shape[0]))
                f.write(struct.pack('I', W_i.shape[1]))
                f.write(struct.pack('I', U_i.shape[0]))
                f.write(struct.pack('I', U_i.shape[1]))
                f.write(struct.pack('I', b_i.shape[0]))

                f.write(struct.pack('I', W_f.shape[0]))
                f.write(struct.pack('I', W_f.shape[1]))
                f.write(struct.pack('I', U_f.shape[0]))
                f.write(struct.pack('I', U_f.shape[1]))
                f.write(struct.pack('I', b_f.shape[0]))

                f.write(struct.pack('I', W_c.shape[0]))
                f.write(struct.pack('I', W_c.shape[1]))
                f.write(struct.pack('I', U_c.shape[0]))
                f.write(struct.pack('I', U_c.shape[1]))
                f.write(struct.pack('I', b_c.shape[0]))

                f.write(struct.pack('I', W_o.shape[0]))
                f.write(struct.pack('I', W_o.shape[1]))
                f.write(struct.pack('I', U_o.shape[0]))
                f.write(struct.pack('I', U_o.shape[1]))
                f.write(struct.pack('I', b_o.shape[0]))

                W_i = W_i.flatten()
                U_i = U_i.flatten()
                b_i = b_i.flatten()
                W_f = W_f.flatten()
                U_f = U_f.flatten()
                b_f = b_f.flatten()
                W_c = W_c.flatten()
                U_c = U_c.flatten()
                b_c = b_c.flatten()
                W_o = W_o.flatten()
                U_o = U_o.flatten()
                b_o = b_o.flatten()

                write_floats(f, W_i)
                write_floats(f, U_i)
                write_floats(f, b_i)
                write_floats(f, W_f)
                write_floats(f, U_f)
                write_floats(f, b_f)
                write_floats(f, W_c)
                write_floats(f, U_c)
                write_floats(f, b_c)
                write_floats(f, W_o)
                write_floats(f, U_o)
                write_floats(f, b_o)

                write_activation(inner_activation)
                write_activation(activation)
                f.write(struct.pack('I', return_sequences))

            elif layer_type == 'Embedding':
                weights = layer.get_weights()[0]

                f.write(struct.pack('I', LAYER_EMBEDDING))
                f.write(struct.pack('I', weights.shape[0]))
                f.write(struct.pack('I', weights.shape[1]))

                weights = weights.flatten()

                write_floats(f, weights)

            else:
                assert False, "Unsupported layer type: %s" % layer_type
