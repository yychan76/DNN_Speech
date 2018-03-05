from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bn_input_rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add batch normalization 
    bn_input = BatchNormalization(name='bn_input')(input_data)
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(bn_input)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(simp_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif border_mode == 'causal':
        output_length = input_length
    return (output_length + stride - 1) // stride

def dilated_cnn_rnn_model(input_dim, filters, kernel_size, conv_border_mode,
                          units, conv_layers, dropout=0.0, recurrent_dropout=0.0, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Current model stride value != 1 is incompatible with specifying any dilation_rate value != 1
    conv_stride = 1
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    cnn = input_data
    dilation_rates = [1, 2, 4, 8, 16]
    for i in range(conv_layers):
        # Add convolutional layer
        cnn = Conv1D(filters, kernel_size, 
                        strides=conv_stride, 
                        padding=conv_border_mode,
                        dilation_rate=dilation_rates[i % len(dilation_rates)],
                        activation='relu',
                        name='conv1d{}'.format(str(i+1)))(cnn)
        # Add Dropout
        cnn = Dropout(dropout, name='cnn_dropout{}'.format(str(i+1)))(cnn)
        # Add batch normalization
        cnn = BatchNormalization(name='bn_conv1d{}'.format(str(i+1)))(cnn)
    # Add a recurrent layer
    rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=1, dropout=dropout, recurrent_dropout=recurrent_dropout, name='rnn')(cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Calculate output_length
    def get_output_length(x):
        for i in range(conv_layers):
            x = cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, 
                                  dilation=dilation_rates[i % len(dilation_rates)])
        return x

    model.output_length = lambda x: get_output_length(x)
    print(model.summary())
    return model

def deep_rnn_model(input_dim, units, recur_layers, activation='relu', output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn = input_data
    for i in range(recur_layers):
        rnn = GRU(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn{}'.format(str(i+1)))(rnn)
        # Add batch normalization 
        rnn = BatchNormalization(name='bn_rnn{}'.format(str(i+1)))(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, activation='relu', output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation=activation,
        return_sequences=True, implementation=2), merge_mode='concat', name='bidir_rnn')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deep_bidirectional_rnn_model(input_dim, units, recur_layers, activation='relu', output_dim=29):
    """ Build a deep bidirectional recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn = input_data
    for i in range(recur_layers):
        rnn = Bidirectional(GRU(units, activation=activation,
            return_sequences=True, implementation=2), merge_mode='concat', name='bidir_rnn{}'.format(str(i+1)))(rnn)
        # Add batch normalization 
        rnn = BatchNormalization(name='bn_rnn{}'.format(str(i+1)))(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deep_bidirectional_rnn_model_lessmem(input_dim, units, recur_layers, activation='relu', output_dim=29):
    """ Build a deep bidirectional recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn = input_data
    for i in range(recur_layers):
        rnn = Bidirectional(GRU(units, activation=activation,
            return_sequences=True, implementation=1), merge_mode='concat', name='bidir_rnn{}'.format(str(i+1)))(rnn)
        # Add batch normalization 
        rnn = BatchNormalization(name='bn_rnn{}'.format(str(i+1)))(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deep_bidirectional_rnn_model_lessmem_dropout(input_dim, units, recur_layers, activation='relu', dropout=0.0, recurrent_dropout=0.0, output_dim=29):
    """ Build a deep bidirectional recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn = input_data
    for i in range(recur_layers):
        rnn = Bidirectional(GRU(units, activation=activation,
            return_sequences=True, implementation=1, dropout=dropout, recurrent_dropout=recurrent_dropout), merge_mode='concat', name='bidir_rnn{}'.format(str(i+1)))(rnn)
        # Add batch normalization 
        rnn = BatchNormalization(name='bn_rnn{}'.format(str(i+1)))(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim=161, units=200, filters=200, kernel_size=11, conv_border_mode='valid',
                          conv_layers=3, recur_layers=2, activation='relu', dropout=0.0, recurrent_dropout=0.0, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Current model stride value != 1 is incompatible with specifying any dilation_rate value != 1
    conv_stride = 1
    cnn = input_data
    dilation_rates = [1, 2, 4, 8, 16]
    # Add convolutional layers, each with batch normalization and dropout
    for i in range(conv_layers):
        cnn = Conv1D(filters, kernel_size, 
                        strides=conv_stride, 
                        padding=conv_border_mode,
                        dilation_rate=dilation_rates[i % len(dilation_rates)],
                        activation='relu',
                        name='conv1d{}'.format(str(i+1)))(cnn)
        # Add Dropout
        cnn = Dropout(dropout, name='cnn_dropout{}'.format(str(i+1)))(cnn)
        # Add batch normalization
        cnn = BatchNormalization(name='bn_conv1d{}'.format(str(i+1)))(cnn)

    # Add recurrent layers, each with batch normalization
    rnn = cnn
    for i in range(recur_layers):
        rnn = Bidirectional(GRU(units, activation=activation,
            return_sequences=True, implementation=1, dropout=dropout, recurrent_dropout=recurrent_dropout), merge_mode='concat', name='bidir_rnn{}'.format(str(i+1)))(rnn)
        # Add batch normalization 
        rnn = BatchNormalization(name='bn_rnn{}'.format(str(i+1)))(rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    def get_output_length(x):
        for i in range(conv_layers):
            x = cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, 
                                  dilation=dilation_rates[i % len(dilation_rates)])
        return x
    # TODO: Specify model.output_length. output length needs to be known in order to compute CTC loss
    model.output_length = lambda x: get_output_length(x)
    print(model.summary())
    return model