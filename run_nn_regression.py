""" Fine tune BERT models for building regression model
    to predict challenge prize.
"""

import os
import json
import re
from pprint import pprint
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd

import tensorflow as tf
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TFAutoModel,
    TFDistilBertModel,
    TFDistilBertForSequenceClassification,
    TFTrainer,
    TFTrainingArguments
)
from sklearn.metrics import max_error, mean_absolute_error, median_absolute_error, mean_squared_error, r2_score

from tc_data import TopCoder
from model_tcpm_distilbert import build_tcpm_model_distilbert_regression, TCPMDistilBertRegression

load_dotenv()

def compute_metrics(pred):
    """ Compute eval metrics
        reference: https://huggingface.co/transformers/training.html#tensorflow
    """
    y_true = pred.label_ids
    y_pred = pred.predictions
    return {
        'max_err': max_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'medianae': median_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

def mre(y_true, y_pred):
    """ Self defined Mean Relative Error metrics"""
    return tf.math.reduce_mean(tf.math.abs(y_true - y_pred) / y_true)

def run_bert_regression_trainer():
    """ Run bert single class classification(a.k.a regression) model."""
    print('START TRAINNING FOR REGRESSION')
    log_dir = os.path.join(os.getenv('OUTPUT_DIR'), 'hf_trainer')

    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL_NAME'))
    config = AutoConfig.from_pretrained(os.getenv('MODEL_NAME'), num_labels=1)

    training_args = TFTrainingArguments(
        output_dir=log_dir,
        logging_dir=log_dir,
        logging_first_step=True,
        logging_steps=1,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        debug=True,
    )

    with training_args.strategy.scope():
        # model = TFDistilBertForSequenceClassification.from_pretrained(os.getenv('MODEL_NAME'), config=config)
        model = TCPMDistilBertRegression.from_pretrained(os.getenv('MODEL_NAME'), config=config)

    print('\nModel Config:')
    print(config)
    print('Tokenizer: ', tokenizer)
    print('Model: ', model)
    print('\nTFTraingArguments:')
    print(training_args)

    tc = TopCoder()
    encoded_text = tc.get_bert_encoded_txt_features(tokenizer)
    metadata = tc.get_meta_data_features(encoded_tech=True, softmax_tech=True)
    target = tc.get_target()

    split = int((4 / 5) * len(target))
    dataset = tf.data.Dataset.from_tensor_slices((dict(**encoded_text, meta_input=metadata), target))
    dataset = dataset.shuffle(len(target))
    train_ds, test_ds = dataset.take(split), dataset.skip(split)

    print('\nTrain dataset samples:')
    for el in train_ds.take(3):
        pprint(el)
    print('\nTest dataset samples:')
    for el in test_ds.take(3):
        pprint(el)

    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(log_dir)

    result = trainer.evaluate()
    print('\nTrainning eval:')
    pprint(result)
    with open(os.path.join(log_dir, 'eval_results.json'), 'w') as fwrite:
        json.dump(result, fwrite, indent=4)

def run_bert_regression_tfmodel():
    """ Run BERT for regression as a tfmodel."""
    print('START TRAINNING FOR REGRESSION')

    # Initialize BERT model
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL_NAME'))
    config = AutoConfig.from_pretrained(os.getenv('MODEL_NAME'), num_labels=1)
    model = TFDistilBertForSequenceClassification.from_pretrained(os.getenv('MODEL_NAME'), config=config)

    print('\nModel Config:')
    print(config)
    print('Tokenizer: ', tokenizer)
    print('Model: ', model)

    # Preparing training data
    tc = TopCoder()
    encoded_text = tc.get_bert_encoded_txt_features(tokenizer)
    target = tc.get_target()

    print(f'\nSize of dataset: {len(target)}')

    dataset = tf.data.Dataset.from_tensor_slices((encoded_text, target))
    dataset = dataset.shuffle(len(target))
    train_ds, test_ds = dataset.take(int((4 / 5) * len(target))), dataset.skip(int((4 / 5) * len(target)))
    train_ds = train_ds.batch(16)
    test_ds = test_ds.batch(8)

    print('\nTrain dataset samples:')
    for el in train_ds.take(3):
        pprint(el)
    print('\nTest dataset samples:')
    for el in test_ds.take(3):
        pprint(el)

    # TF-Fashioned training model
    log_dir = os.path.join(os.getenv('OUTPUT_DIR'), 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) # fancy visulization :)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6),
        loss='mse',
        metrics=['mae', 'mse', mre]
    )
    history = model.fit(
        train_ds,
        epochs=6,
        callbacks=[tensorboard_cb]
    )
    result = model.evaluate(
        test_ds,
        return_dict=True
    )

    pprint(result)

    history_df = pd.DataFrame(history.history)
    history_df.to_json(os.path.join(log_dir, 'train_history.json'), orient='index', indent=4)
    with open(os.path.join(log_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4)

def run_bert_meta_regression_tfmodel():
    """ Run self defined combined model."""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.getenv('OUTPUT_DIR'), timestamp)
    model_plot = f'regression_model_{timestamp}.png'

    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL_NAME'))
    config = AutoConfig.from_pretrained(os.getenv('MODEL_NAME'), num_labels=1)
    distilebert_model = TFDistilBertModel.from_pretrained(os.getenv('MODEL_NAME'), config=config)

    print(config, tokenizer, sep='\n')
    # tf.keras.utils.plot_model(distilebert_model, to_file=model_plot, show_shapes=True)

    tc = TopCoder()
    encoded_text = tc.get_bert_encoded_txt_features(tokenizer)
    metadata = tc.get_meta_data_features(encoded_tech=True, softmax_tech=True)
    target = tc.get_target()

    split = int((4 / 5) * len(target))
    dataset = tf.data.Dataset.from_tensor_slices((dict(**encoded_text, meta_input=metadata), target))
    dataset = dataset.shuffle(len(target))
    train_ds, test_ds = dataset.take(split).batch(16), dataset.skip(split).batch(8)

    print(train_ds, test_ds, sep='\n')
    # for i in train_ds.take(2):
    #     pprint(i)
    # print()
    # for i in test_ds.take(2):
    #     pprint(i)

    # model = TCPMDistilBertRegression.from_pretrained(os.getenv('MODEL_NAME'), config=config)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model = build_tcpm_model_distilbert_regression(distilebert_model)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-6),
        loss='mse',
        metrics=['mae', 'mse', mre]
    )
    history = model.fit(
        train_ds,
        epochs=12,
    )
    result = model.evaluate(
        test_ds,
        return_dict=True,
    )

    pprint(result)

    history_df = pd.DataFrame(history.history)
    history_df.to_json(os.path.join(log_dir, 'train_history.json'), orient='index', indent=4)
    with open(os.path.join(log_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    # run_metadata_model()
    # run_bert_regression_tfmodel()
    # run_bert_regression_trainer()
    run_bert_meta_regression_tfmodel()
