""" A Neural Network model that takes both metadata and requirement text
    as input to classify which prize range this challenge falls into.
"""
import os
import json

from dotenv import load_dotenv

import numpy as np
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TFDistilBertPreTrainedModel, # TFBertPreTrainedModel,
    TFDistilBertMainLayer, # TFBertMainLayer,
    TFDistilBertModel, # TFBertModel
)
from transformers.modeling_tf_utils import (
    TFSequenceClassificationLoss,
    get_initializer,
    shape_list
)

DUMMY_META = tf.cast(tf.constant(np.random.randint(1, 1000, (3, 4))), tf.float64)

def custom_loss_fn(y_true, y_pred):
    """ A custom loss computation for debugging"""
    print('y_true: ', tf.cast(y_true, tf.int32))
    print('y_pred: ', tf.cast(y_pred, tf.float32))
    # one_hot_positions = tf.one_hot(y_true, shape_list(y_pred)[1])
    one_hot_positions = tf.one_hot(tf.cast(y_true, tf.int32), shape_list(y_pred)[1])
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_positions, logits=y_pred)
    return tf.reduce_mean(loss)

def build_tcpm_model_distilbert(distilbert_model: TFDistilBertModel, config: AutoConfig):
    """ Build TopCoder Pricing Model(TCPM) using tensorflow functional api
        The builiding process will be very similar to BERT model (using TFBertModel)

        The input format of tcpm_model is expected to be:

        ```{'input_ids': Tensor(shape=(512,)), 'attention_mask': Tensor(shape=(512,)), 'meta_input': Tensor(shape=(4,))}```
    """
    # define layers needed to build the model
    fully_connected_layer = tf.keras.layers.Dense(
        config.dim,
        activation='relu',
        name='fully_connected'
    )
    dropout_layer = tf.keras.layers.Dropout(config.seq_classif_dropout)
    classification = tf.keras.layers.Dense(config.num_labels, name='classification')
    # softmax_layer = tf.keras.layers.Softmax()

    # build (distil)bert model pipeline
    distilbert_input = {k: tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name=k) for k in ('input_ids', 'attention_mask')} # there probably is a better way to get the encoded keys but it works for now
    distilbert_output = distilbert_model(distilbert_input)
    hidden_state = distilbert_output[0] # (batch_size, seq_len, dimension) get last layer hidden state
    pooled_output = hidden_state[:, 0] # (bs, dimension)

    meta_input = tf.keras.layers.Input(shape=(4,), dtype=tf.float32, name='meta_input') # challenge meta input

    # append metadata features to the output of (distil)bert output
    # continue forward to the fully connected layer
    concat_layer = tf.keras.layers.concatenate([pooled_output, meta_input])
    x = fully_connected_layer(concat_layer)
    # x = dropout_layer(x)
    output = classification(x)
    # output = softmax_layer(output)

    tcpm_model = tf.keras.Model(inputs=[distilbert_input, meta_input], outputs=output)
    return tcpm_model

class TCPMDistilBertClassification(TFDistilBertPreTrainedModel, TFSequenceClassificationLoss):
    """ Classification model that takes both encoded text and metadata as input."""
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.distilbert = TFDistilBertMainLayer(config, name='distilbert')
        self.metadata_inputs = tf.keras.layers.InputLayer(input_shape=(4,), name='metadata')
        self.fully_connected = tf.keras.layers.Dense(
            config.dim,
            kernel_initializer=get_initializer(config.initializer_range),
            activation='relu',
            name='fully_connected',
        )
        self.dropout = tf.keras.layers.Dropout(config.seq_classif_dropout)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name='classifier'
        )

    def call(self, inputs=None, labels=None, training=False):
        """ The inputs has type of shape as follow:
            ({'input_ids': (512,), 'attention_mask': (512,)}, (4,))
            which are (encoded_text, metadata)
        """
        if isinstance(inputs, (tuple, list)):
            bert_inputs, metadata_inputs = inputs
        else:
            bert_inputs, metadata_inputs = inputs, DUMMY_META

        distilbert_output = self.distilbert(bert_inputs, training=training)
        hidden_state = distilbert_output[0] # (bs, seq_len, dim)

        # append metadata after the bert output embeddings
        pooled_output = hidden_state[:, 0] # (bs, dim) gen sentence embedding == embedding of '[CLS]'
        metadata_output = self.metadata_inputs(metadata_inputs)
        concat_output = tf.keras.layers.concatenate([pooled_output, metadata_output])

        # continue forward
        x = self.fully_connected(concat_output)
        # x = self.dropout(x, training=training)
        logits = self.classifier(x)

        outputs = (logits,) + distilbert_output[1:] # copy-paste from original impolementation

        if labels is not None:
            loss = self.compute_loss(labels, logits)
            outputs = (loss,) + outputs

        return outputs # (loss), logits, (hidden_states), (attentions)
