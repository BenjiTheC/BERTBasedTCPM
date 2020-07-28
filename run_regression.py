""" Fine tune BERT models for building regression model
    to predict challenge prize.
"""

import os
import json
import re
from pprint import pprint
from dotenv import load_dotenv

import numpy as np
import pandas as pd

import tensorflow as tf
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TFAutoModel,
    TFDistilBertForSequenceClassification,
    TFTrainer,
    TFTrainingArguments
)
from sklearn.metrics import max_error, mean_absolute_error, median_absolute_error, mean_squared_error, r2_score

from tc_data import TopCoder

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

def run_bert_regression():
    """ Run bert single class classification(a.k.a regression) model."""
    print('START TRAINNING FOR REGRESSION')

    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL_NAME'))
    config = AutoConfig.from_pretrained(os.getenv('MODEL_NAME'), num_labels=1)

    training_args = TFTrainingArguments(
        output_dir=os.getenv('OUTPUT_DIR'),
        logging_dir=os.path.join(os.getenv('OUTPUT_DIR'), 'log'),
        logging_first_step=True,
        logging_steps=1,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        debug=True,
    )

    with training_args.strategy.scope():
        model = TFDistilBertForSequenceClassification.from_pretrained(os.getenv('MODEL_NAME'), config=config)

    print('\nModel Config:')
    print(config)
    print('Tokenizer: ', tokenizer)
    print('Model: ', model)
    print('\nTFTraingArguments:')
    print(training_args)

    tc = TopCoder()
    encoded_text = tc.get_bert_encoded_txt_features(tokenizer)
    target = tc.get_target()

    print(f'\nSize of dataset: {len(target)}')

    dataset = tf.data.Dataset.from_tensor_slices((encoded_text, target))
    dataset = dataset.shuffle(len(target))
    train_ds, test_ds = dataset.take(int((4 / 5) * len(target))), dataset.skip(int((4 / 5) * len(target)))

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
    tokenizer.save_pretrained(os.getenv('OUTPUT_DIR'))

    result = trainer.evaluate()
    print('\nTrainning eval:')
    pprint(result)
    with open(os.path.join(os.getenv('OUTPUT_DIR'), 'eval_results.json'), 'w') as fwrite:
        json.dump(result, fwrite, indent=4)

if __name__ == "__main__":
    run_bert_regression()
