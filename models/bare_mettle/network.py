""" Network wiring """

import tensorflow as tf
import numpy as np
import glob, time, os
from diagnostics import Diagnostics

class Network(object):

    @staticmethod
    def birnn(x, config, training, attention=False):
         # reshape outputs to [batch_size, max_time_steps, n_features]
        max_time = tf.shape(x)[1]
        rnn_inputs = tf.reshape(x, [-1, max_time, config.embedding_dim])
        sequence_lengths = Utils.length(rnn_inputs)
        init = tf.contrib.layers.xavier_initializer()

         # Choose rnn cell type
        if config.rnn_cell == 'lstm':
            args = {'num_units': config.hidden_units, 'forget_bias': 1.0, 'state_is_tuple': True}
            base_cell = tf.nn.rnn_cell.LSTMCell
        elif config.rnn_cell == 'gru':
            args = {'num_units': config.hidden_units}
            base_cell = tf.nn.rnn_cell.GRUCell
        elif config.rnn_cell == 'layer-norm':
            args = {'num_units': config.hidden_units, 'forget_bias': 1.0, 'dropout_keep_prob': self.config.recurrent_keep_prob}
            base_cell = tf.contrib.rnn.LayerNormBasicLSTMCell
     
        cell = base_cell

        if training and config.output_keep_prob < 1:
            # rnn_inputs = tf.nn.dropout(rnn_inputs, self.keep_prob)
            fwd_cells = [tf.nn.rnn_cell.DropoutWrapper(
                cell(**args), 
                output_keep_prob=config.output_keep_prob,
                variational_recurrent=True) for _ in range(config.rnn_layers)]
            bwd_cells = [tf.nn.rnn_cell.DropoutWrapper(
                cell(**args),
                output_keep_prob=config.output_keep_prob,
                variational_recurrent=True) for _ in range(config.rnn_layers)]
        else:
            fwd_cells = [cell(**args) for _ in range(config.rnn_layers)]
            bwd_cells = [cell(**args) for _ in range(config.rnn_layers)]
        
        birnn_output, fwd_state, bwd_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=fwd_cells,
            cells_bw=bwd_cells,
            inputs=rnn_inputs,
            sequence_length=sequence_lengths,
            parallel_iterations=128)

        if attention:  # invoke soft attention mechanism - attend to different particles
            summary_vector = attention_A(birnn_output, config.attention_dim, my_method=False)
        else:  # Select last relevant output
            summary_vector = Diagnostics.last_relevant(birnn_output, sequence_lengths)

        # Fully connected layer for classification
        with tf.variable_scope("fc"):
            logits_RNN = tf.layers.dense(summary_vector, units=config.n_classes, kernel_initializer=init)
        
        smx, pred = tf.nn.softmax(logits_RNN), tf.argmax(logits_RNN, 1)

        return logits_RNN, smx, pred       

    @staticmethod
    def wrn(x, config, training, reuse=False, actv=tf.nn.relu):
        # Implements W-28-10 wide residual network
        # See Arxiv 1605.07146
        network_width = 10 # k
        block_multiplicity = 3 # n

        filters = [16, 16, 32, 64]
        init = tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':True}

        def residual_block(x, n_filters, actv, keep_prob, training, project_shortcut=False, first_block=False):
            init = tf.contrib.layers.xavier_initializer()
            kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':True}

            if project_shortcut:
                strides = [2,2] if not first_block else [1,1]
                identity_map = tf.layers.conv2d(x, filters=n_filters, kernel_size=[1,1],
                                   strides=strides, kernel_initializer=init, padding='same')
                # identity_map = tf.layers.batch_normalization(identity_map, **kwargs)
            else:
                strides = [1,1]
                identity_map = x

            bn = tf.layers.batch_normalization(x, **kwargs)
            conv = tf.layers.conv2d(bn, filters=n_filters, kernel_size=[3,3], activation=actv,
                       strides=strides, kernel_initializer=init, padding='same')

            bn = tf.layers.batch_normalization(conv, **kwargs)
            do = tf.layers.dropout(bn, rate=1-keep_prob, training=training)

            conv = tf.layers.conv2d(do, filters=n_filters, kernel_size=[3,3], activation=actv,
                       kernel_initializer=init, padding='same')
            out = tf.add(conv, identity_map)

            return out

        def residual_block_2(x, n_filters, actv, keep_prob, training, project_shortcut=False, first_block=False):
            init = tf.contrib.layers.xavier_initializer()
            kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':True}
            prev_filters = x.get_shape().as_list()[-1]
            if project_shortcut:
                strides = [2,2] if not first_block else [1,1]
                # identity_map = tf.layers.conv2d(x, filters=n_filters, kernel_size=[1,1],
                #                   strides=strides, kernel_initializer=init, padding='same')
                identity_map = tf.layers.average_pooling2d(x, strides, strides, 'valid')
                identity_map = tf.pad(identity_map, 
                    tf.constant([[0,0],[0,0],[0,0],[(n_filters-prev_filters)//2, (n_filters-prev_filters)//2]]))
                # identity_map = tf.layers.batch_normalization(identity_map, **kwargs)
            else:
                strides = [1,1]
                identity_map = x

            x = tf.layers.batch_normalization(x, **kwargs)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=n_filters, kernel_size=[3,3], strides=strides,
                    kernel_initializer=init, padding='same')

            x = tf.layers.batch_normalization(x, **kwargs)
            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, rate=1-keep_prob, training=training)

            x = tf.layers.conv2d(x, filters=n_filters, kernel_size=[3,3],
                       kernel_initializer=init, padding='same')
            out = tf.add(x, identity_map)

            return out

        with tf.variable_scope('wrn_conv', reuse=reuse):
            # Initial convolution --------------------------------------------->
            with tf.variable_scope('conv0', reuse=reuse):
                conv = tf.layers.conv2d(x, filters[0], kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
            # Residual group 1 ------------------------------------------------>
            rb = conv
            f1 = filters[1]*network_width
            for n in range(block_multiplicity):
                with tf.variable_scope('group1/{}'.format(n), reuse=reuse):
                    project_shortcut = True if n==0 else False
                    rb = residual_block(rb, f1, actv, project_shortcut=project_shortcut,
                            keep_prob=config.conv_keep_prob, training=training, first_block=True)
            # Residual group 2 ------------------------------------------------>
            f2 = filters[2]*network_width
            for n in range(block_multiplicity):
                with tf.variable_scope('group2/{}'.format(n), reuse=reuse):
                    project_shortcut = True if n==0 else False
                    rb = residual_block(rb, f2, actv, project_shortcut=project_shortcut,
                            keep_prob=config.conv_keep_prob, training=training)
            # Residual group 3 ------------------------------------------------>
            f3 = filters[3]*network_width
            for n in range(block_multiplicity):
                with tf.variable_scope('group3/{}'.format(n), reuse=reuse):
                    project_shortcut = True if n==0 else False
                    rb = residual_block(rb, f3, actv, project_shortcut=project_shortcut,
                            keep_prob=config.conv_keep_prob, training=training)
            # Avg pooling + output -------------------------------------------->
            with tf.variable_scope('output', reuse=reuse):
                bn = tf.nn.relu(tf.layers.batch_normalization(rb, **kwargs))
                avp = tf.layers.average_pooling2d(bn, pool_size=[8,8], strides=[1,1], padding='valid')
                flatten = tf.contrib.layers.flatten(avp)
                out = tf.layers.dense(flatten, units=config.n_classes, kernel_initializer=init)

            return out