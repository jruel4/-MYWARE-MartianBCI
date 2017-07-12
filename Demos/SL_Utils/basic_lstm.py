from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers


def lstm_model(num_units, rnn_layers, dense_layers=None, learning_rate=0.001, optimizer='Momentum'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(
                    layer['num_units'], state_is_tuple=True
                ),
                layer['keep_prob']
            ) if layer.get('keep_prob') else tf.contrib.rnn.BasicLSTMCell(
                    layer['num_units'],
                    state_is_tuple=True
                ) for layer in layers
            ]
        return [tf.contrib.rnn.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unstack(X, axis=1, num=num_units)
        output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)
        
        # Delete this line
        #prediction, loss = tflearn.models.linear_regression(output, y)
        
        #prediction = tflayers.softmax(output)
        prediction = tf.argmax(output,1)
        #one_hot_y = tf.one_hot(y, dense_layers[0])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y,name = 'softmax'))

        #ex_func = lambda lr,gs : tf.train.exponential_decay(lr, gs, 250, .96)
        
        mom_lambda = lambda lr: tf.train.RMSPropOptimizer(lr)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=mom_lambda,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model













