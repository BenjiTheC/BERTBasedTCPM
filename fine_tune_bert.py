""" This script fine-tunes BERT model
    Using `HfArgumentParser` to construct a CLI so it's easily repeated with different params.

    Largely inspired by the 🤗transformers example for running GLUE task:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_tf_glue.py
"""

import os
import json
import re
from pprint import pprint
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Union

import numpy as np
import pandas as pd

import tensorflow as tf
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TFAutoModel,
    TFAutoModelForSequenceClassification,
    TFBertPreTrainedModel,
    TFTrainer,
    TFTrainingArguments,
)
from sklearn.metrics import precision_recall_fscore_support

from tc_data import TopCoder
from model_tcpm_distilbert import TCPMDistilBertClassification, build_tcpm_model_distilbert, custom_loss_fn

load_dotenv()
# MODEL_NAME = os.getenv('MODEL_NAME')
# OUTPUT_DIR = os.getenv('OUTPUT_DIR')

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4

def build_dataset(tokenizer):
    """ Build td.data.Dataset out of text and prize range."""
    # Load TopCoder data
    tc = TopCoder()
    tc_req = tc.get_filtered_requirements()
    tc_meta = tc.get_filtered_challenge_info()
    metadata_cols = ['number_of_platforms', 'number_of_technologies', 'project_id', 'challenge_duration']

    # Convert float prize into categorical prize range
    interval = np.linspace(0, 2500, 51)[:-1] # this will be modified manuanly when the dataset is changed
    tc_prz_range = tc_meta['total_prize'].apply(lambda prz: np.searchsorted(interval, prz, side='right') - 1)
    tc_prz_range.name = 'prize_cat'

    # use this df to ensure the index of text and metadata and label is aligned
    req_prz_df = pd.concat([
        tc_req['requirement'],
        tc_meta.reindex(metadata_cols, axis=1),
        tc_prz_range
        ], axis=1)

    dataset_size = len(req_prz_df)
    num_labels = len(req_prz_df['prize_cat'].unique()) + 1

    # batched encode the str to `input_ids` and `attention_mask`
    batched_encoded = tokenizer(req_prz_df['requirement'].to_list(), padding='max_length', truncation=True)

    # List((enccoded_str, metadata, prize_cat),...)
    features = [(
        {k: batched_encoded[k][i] for k in batched_encoded},
        req_prz_df.reindex(metadata_cols, axis=1).iloc[i],
        req_prz_df['prize_cat'].iloc[i]
        ) for i in range(len(req_prz_df))]

    def gen():
        """ generator used in `tf.data.Dataset.from_generator`."""
        for encoded_str, metadata, label in features:
            yield dict(**encoded_str, meta_input=metadata), label # NOTE: it's import the key if named "meta_input" to match the input_layer's name in the model

    dataset = tf.data.Dataset.from_generator(
        gen,
        (
            dict(**{k: tf.int32 for k in batched_encoded}, meta_input=tf.float32),
            tf.int32
            ),
        (
            dict(**{k: tf.TensorShape([512]) for k in batched_encoded}, meta_input=tf.TensorShape([4])),
            tf.TensorShape([])
            )
    )

    return (
        dataset,
        dataset_size,
        num_labels
    )

def compute_metrics(pred):
    """ Compute eval metrics
        reference: https://huggingface.co/transformers/training.html#tensorflow
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = (preds == labels).mean()
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def finetune_with_tftrainer():
    """ Fine tune with TFTrainer but it's not working with some package error
        Found an issue from github of the same problem I have, follow up there:
        https://github.com/huggingface/transformers/issues/5151
    """
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL_NAME'))

    # Get data for fine-tuning
    dataset, dataset_size, num_labels = build_dataset(tokenizer)

    config = AutoConfig.from_pretrained(os.getenv('MODEL_NAME'), num_labels=num_labels)

    training_args = TFTrainingArguments(
        output_dir=os.getenv('OUTPUT_DIR'),
        logging_dir=os.getenv('OUTPUT_DIR'),
        logging_first_step=True,
        logging_steps=1,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        debug=True
        )

    with training_args.strategy.scope():
        # model = TFAutoModel.from_pretrained(os.getenv('MODEL_NAME'), config=config, cache_dir=os.getenv('OUTPUT_DIR'))
        model = TCPMDistilBertClassification.from_pretrained(os.getenv('MODEL_NAME'), config=config)

    # shuffle and split train/test tasks manuanly
    dataset = dataset.shuffle(dataset_size)
    train_size = int(dataset_size * (4 / 5))
    # test_size = dataset_size - train_size # didn't use so commented out.

    train_data = dataset.take(train_size)#.batch(TRAIN_BATCH_SIZE)
    test_data = dataset.skip(train_size)#.batch(TEST_BATCH_SIZE)

    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics
    )

    print('Sample training data:')
    for i in train_data.take(2):
        print(f'Input #{i}:')
        pprint(i[0])
        print(f'Target #{i}:')
        pprint(i[1])

    print('Sample test data:')
    for i in train_data.take(2):
        print(f'Input #{i}:')
        pprint(i[0])
        print(f'Output #{i}:')
        pprint(i[1])

    print('TF Trainer args:')
    pprint(training_args)
    print('TCPM Model config:')
    pprint(config)
    print('Trainning start..')

    # Train the model
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(os.getenv('OUTPUT_DIR'))

    # Evaluate the model
    result = trainer.evaluate()
    pprint(result)
    with open(os.path.join(os.getenv('OUTPUT_DIR'), 'eval_results.json'), 'w') as fwrite:
        json.dump(result, fwrite, indent=4)

def finetune_tcpm_as_tfmodel():
    """ Fine tune the subclass model as a tf model"""
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL_NAME'))

    # Get data for fine-tuning
    dataset, dataset_size, num_labels = build_dataset(tokenizer)

    config = AutoConfig.from_pretrained(os.getenv('MODEL_NAME'), num_labels=num_labels)
    model = TCPMDistilBertClassification.from_pretrained(os.getenv('MODEL_NAME'), config=config)

    # shuffle and split train/test tasks manuanly
    dataset = dataset.shuffle(dataset_size)
    train_size = int(dataset_size * (4 / 5))
    # test_size = dataset_size - train_size # didn't use so commented out.

    train_data = dataset.take(train_size).batch(TRAIN_BATCH_SIZE)
    test_data = dataset.skip(train_size).batch(TEST_BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True), 'accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(train_data, epochs=2)
    result = model.evaluate(test_data)
    print(result)

def finetune_tf_function():
    """ Fine tune functional api."""
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL_NAME'))

    # Get data for fine-tuning
    dataset, dataset_size, num_labels = build_dataset(tokenizer)
    # shuffle and split train/test tasks manuanly
    dataset = dataset.shuffle(dataset_size)
    train_size = int(dataset_size * (4 / 5))
    # test_size = dataset_size - train_size # didn't use so commented out.
    train_data = dataset.take(train_size).batch(TRAIN_BATCH_SIZE)
    test_data = dataset.skip(train_size).batch(TEST_BATCH_SIZE)

    print('Sample training data:')
    for i in train_data.take(2):
        print('Input: ', end='')
        pprint(i[0])
        print('Target: ', end='')
        pprint(i[1])

    print('Sample test data:')
    for i in train_data.take(2):
        print('Input: ', end='')
        pprint(i[0])
        print('Output: ', end='')
        pprint(i[1])

    config = AutoConfig.from_pretrained(os.getenv('MODEL_NAME'), num_labels=num_labels)
    distilbert_model = TFAutoModel.from_pretrained(os.getenv('MODEL_NAME'), config=config)

    model = build_tcpm_model_distilbert(distilbert_model=distilbert_model, config=config)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True), 'accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # train the model
    train_history = model.fit(train_data, epochs=4)
    print('Training finish:')
    pprint(train_history)
    # evaluate the model
    result = model.evaluate(test_data)
    print('Test result:')
    pprint(result)

if __name__ == "__main__":
    # finetune_with_tftrainer()
    # finetune_tf_function()
    finetune_tcpm_as_tfmodel()
