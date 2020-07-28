""" A Neural Network model that takes both metadata and requirement text
    as input to classify which prize range this challenge falls into.
"""
import os
import json
import random

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
from transformers.tokenization_utils import BatchEncoding

def build_tcpm_model_distilbert_classification(distilbert_model: TFDistilBertModel, config: AutoConfig):
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

def build_tcpm_model_distilbert_regression(distilbert_model: TFDistilBertModel):
    """ Build TCPM regression nn model."""
    fully_connected_layer = tf.keras.layers.Dense(512, activation='relu', name='fully_connected')
    regression_unit = tf.keras.layers.Dense(1, name='regression')

    # build input for DistilBERT and metadata.
    distilbert_input = {k: tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name=k) for k in ('input_ids', 'attention_mask')}
    meta_input = tf.keras.Input(shape=(35,), dtype=tf.float32, name='meta_input')

    # distilBERT output
    distilbert_output = distilbert_model(distilbert_input)
    hidden_state = distilbert_output[0] # (batch_size, seq_len, dimension)
    pooled_output = hidden_state[:, 0] # (bs, dimension)

    # concat distilBERT output and metainput
    concat_layer = tf.keras.layers.concatenate([pooled_output, meta_input], name='concat_bert_meta')
    x = fully_connected_layer(concat_layer)
    output = regression_unit(x)

    model = tf.keras.Model(inputs=[distilbert_input, meta_input], outputs=output)
    return model

class TCPMDistilBertClassification(TFDistilBertPreTrainedModel, TFSequenceClassificationLoss):
    """ Classification model that takes both encoded text and metadata as input."""
    @property
    def dummy_inputs(self):
        """ Overt write the parent class dummy inputs
            used to build the network
        """
        meaningful_encoded_digits = random.randint(12, 512)
        return {
            'input_ids': tf.constant(
                np.concatenate(
                    (
                        np.random.randint(1, 1000, (3, meaningful_encoded_digits)),
                        np.zeros((3, 512 - meaningful_encoded_digits))
                    ), axis=1), dtype=tf.int32),
            'attention_mask': tf.constant(
                np.concatenate(
                    (
                        np.ones((3, meaningful_encoded_digits)),
                        np.zeros((3, 512 - meaningful_encoded_digits))
                    ), axis=1), dtype=tf.int32),
            'meta_input': tf.constant(np.random.randint(1, 1000, (3, 4)), dtype=tf.float32)
        }


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

    def call(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        training=False,
    ):
        """ The inputs has type of shape as follow:
            ({'input_ids': (512,), 'attention_mask': (512,), 'meta_input': (4,)})
            which are (encoded_text, metadata)
        """
        if isinstance(inputs, (tuple, list)):
            labels = inputs[6] if len(inputs) > 6 else labels
            if len(inputs) > 6:
                inputs = inputs[:6]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop('labels', labels)
            if 'meta_input' not in inputs:
                raise KeyError(f'You need to include meta_input in the input data to train this model. Keys of input: {list(inputs.keys())}')
            meta_input = inputs.pop('meta_input')

        print('Input TCPM call')
        print(inputs)
        print(meta_input)
        print(labels)

        distilbert_output = self.distilbert(
            inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            training=training,
        )
        hidden_state = distilbert_output[0] # (bs, seq_len, dim)

        # print('Get hidden state of distilbert', hidden_state)

        # append metadata after the bert output embeddings
        pooled_output = hidden_state[:, 0] # (bs, dim) gen sentence embedding == embedding of '[CLS]'
        metadata_output = self.metadata_inputs(meta_input)
        # print('Get meta output', metadata_output)
        concat_output = tf.keras.layers.concatenate([pooled_output, metadata_output])
        # print('Get concat', concat_output)

        # continue forward
        x = self.fully_connected(concat_output)
        # x = self.dropout(x, training=training)
        logits = self.classifier(x)

        # print('logits', logits)

        outputs = (logits,) + distilbert_output[1:] # copy-paste from original impolementation

        if labels is not None:
            loss = self.compute_loss(labels, logits)
            # print('get loss', loss)
            outputs = (loss,) + outputs

        return outputs # (loss), logits, (hidden_states), (attentions)

class TCPMDistilBertRegression(TFDistilBertPreTrainedModel, TFSequenceClassificationLoss):
    """ Classification model that takes both encoded text and metadata as input."""
    @property
    def dummy_inputs(self):
        """ Overt write the parent class dummy inputs
            used to build the network
        """
        meaningful_encoded_digits = random.randint(12, 512)
        return {
            'input_ids': tf.constant(
                np.concatenate(
                    (
                        np.random.randint(1, 1000, (3, meaningful_encoded_digits)),
                        np.zeros((3, 512 - meaningful_encoded_digits))
                    ), axis=1), dtype=tf.int32),
            'attention_mask': tf.constant(
                np.concatenate(
                    (
                        np.ones((3, meaningful_encoded_digits)),
                        np.zeros((3, 512 - meaningful_encoded_digits))
                    ), axis=1), dtype=tf.int32),
            'meta_input': tf.constant(np.random.randint(1, 1000, (3, 35)), dtype=tf.float32)
        }


    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.distilbert = TFDistilBertMainLayer(config, name='distilbert')
        self.metadata_inputs = tf.keras.layers.InputLayer(input_shape=(35,), name='metadata')
        self.fully_connected = tf.keras.layers.Dense(
            512,
            kernel_initializer=get_initializer(config.initializer_range),
            activation='relu',
            name='fully_connected',
        )
        # self.dropout = tf.keras.layers.Dropout(config.seq_classif_dropout)
        self.regressor = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name='regressor'
        )

    def call(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        training=False,
    ):
        """ The inputs has type of shape as follow:
            ({'input_ids': (512,), 'attention_mask': (512,), 'meta_input': (4,)})
            which are (encoded_text, metadata)
        """
        if isinstance(inputs, (tuple, list)):
            labels = inputs[6] if len(inputs) > 6 else labels
            if len(inputs) > 6:
                inputs = inputs[:6]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop('labels', labels)
            if 'meta_input' not in inputs:
                raise KeyError(f'You need to include meta_input in the input data to train this model. Keys of input: {list(inputs.keys())}')
            meta_input = inputs.pop('meta_input')

        distilbert_output = self.distilbert(
            inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            training=training,
        )
        hidden_state = distilbert_output[0] # (bs, seq_len, dim)

        # print('Get hidden state of distilbert', hidden_state)

        # append metadata after the bert output embeddings
        pooled_output = hidden_state[:, 0] # (bs, dim) gen sentence embedding == embedding of '[CLS]'
        metadata_output = self.metadata_inputs(meta_input)
        concat_output = tf.keras.layers.concatenate([pooled_output, metadata_output])

        # continue forward
        x = self.fully_connected(concat_output)
        # x = self.dropout(x, training=training)
        logits = self.regressor(x)

        outputs = (logits,) + distilbert_output[1:] # copy-paste from original impolementation

        if labels is not None:
            loss = self.compute_loss(labels, logits)
            outputs = (loss,) + outputs

        return outputs # (loss), logits, (hidden_states), (attentions)
