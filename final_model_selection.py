""" The final model selection that compares Gradient Boosting algorithm
    with Neural Network
"""

import os
import json
from pprint import pprint
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, scale, normalize
from sklearn.model_selection import (
    cross_val_predict,
    cross_validate,
    KFold
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    make_scorer
)
from sklearn.pipeline import make_pipeline, Pipeline
from dotenv import load_dotenv

load_dotenv()

def mre(y_true, y_pred):
    """ Calculate the mean relative error of predictions and ground truth."""
    return np.mean(np.abs(y_true - y_pred) / y_true)

def tfmre(y_true, y_pred):
    """ Calculate mre, TensorFlow version for metrics."""
    return tf.math.reduce_mean(tf.math.abs(y_true - y_pred) / y_true)

def build_sequential_neural_network(num_hidden_layers=2, dimension=64, input_shape=(128,)):
    """ Build sequential model with given hidden layer and dimensions."""
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape, name=f'input_layer'),
        *[tf.keras.layers.Dense(dimension, activation='relu', name=f'layer_{i}') for i in range(num_hidden_layers)],
        tf.keras.layers.Dense(1, name='reg'),
    ], name=f'model_h{num_hidden_layers}d{dimension}')

def kfold_predict_validate_gradient_boosting(X: pd.DataFrame, y: pd.Series, cv=10, loss='ls', tol=0.01, n_iter_no_change=5):
    """ Perform K-Fold validation and prediction for gradient boosting."""
    if not all(X.index == y.index):
        raise ValueError('Index of X and y are not equal!')

    kfold = KFold(n_splits=cv)
    cha_id_arr = np.array(X.index)

    Xnp, ynp = X.to_numpy(), y.to_numpy()

    pred_sr_lst = []
    cv_eval_res = []
    for train_idx, test_idx in kfold.split(Xnp):
        X_train, y_train = Xnp[train_idx], ynp[train_idx]
        X_test, y_test = Xnp[test_idx], ynp[test_idx]
        test_cha_id = cha_id_arr[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        gbreg = GradientBoostingRegressor(
            n_estimators=2000,
            loss=loss,
            tol=tol,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=0.2,
            random_state=42,
            verbose=1,
        )
        gbreg.fit(X_train, y_train)

        y_p = gbreg.predict(X_test)
        pred_sr_lst.append(pd.Series(y_p, index=test_cha_id))

        cv_eval_res.append({
            'r2': r2_score(y_test, y_p),
            'mae': mean_absolute_error(y_test, y_p),
            'mse': mean_squared_error(y_test, y_p),
            'mre': mre(y_test, y_p)
        })

    y_pred = pd.concat(pred_sr_lst).reindex(X.index) # algin index with X and y after concat
    overall_score = {
        'r2': r2_score(y, y_pred),
        'mae': mean_absolute_error(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'mre': mre(y, y_pred)
    }

    return y_pred, pd.DataFrame.from_records(cv_eval_res), overall_score

def kfold_predict_validate_neural_network(X: pd.DataFrame, y: pd.Series, cv=10, num_hidden_layer=2, dimension=64, es_min_delta=1):
    """ Perform KFold predict and validation on the whole dataset."""
    if not all(X.index == y.index):
        raise ValueError('Index of X and y are not equal!')

    kfold = KFold(n_splits=cv)
    cha_id_arr = np.array(X.index)

    Xnp, ynp = X.to_numpy(), y.to_numpy()

    pred_sr_lst = []
    cv_eval_res = []

    for train_idx, test_idx in kfold.split(Xnp):
        X_train, y_train = Xnp[train_idx], ynp[train_idx]
        X_test, y_test = Xnp[test_idx], ynp[test_idx]
        test_cha_id = cha_id_arr[test_idx]

        scaler = StandardScaler().fit(X_train)
        normer = Normalizer().fit(X_train)
        X_train = normer.transform(scaler.transform(X_train))
        X_test = normer.transform(scaler.transform(X_test))

        nnreg = build_sequential_neural_network(
            num_hidden_layer,
            dimension,
            input_shape=X.shape[1]
        )
        nnreg.compile(
            optimizer=tf.keras.optimizers.RMSprop(0.0015),
            loss='mse',
            metrics=['mse', 'mae', tfmre],
        )
        escb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=es_min_delta,
            patience=5,
            verbose=1,
        )
        nnreg.fit(
            X_train,
            y_train,
            epochs=500,
            validation_split=0.2,
            batch_size=16,
            callbacks=[escb]
        )

        y_p = nnreg.predict(X_test).reshape(-1)
        pred_sr_lst.append(pd.Series(y_p, index=test_cha_id))

        cv_eval_res.append({
            'r2': r2_score(y_test, y_p),
            'mae': mean_absolute_error(y_test, y_p),
            'mse': mean_squared_error(y_test, y_p),
            'mre': mre(y_test, y_p)
        })

    y_pred = pd.concat(pred_sr_lst).reindex(X.index)
    overall_score = {
        'r2': r2_score(y, y_pred),
        'mae': mean_absolute_error(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'mre': mre(y, y_pred)
    }

    return y_pred, pd.DataFrame.from_records(cv_eval_res), overall_score
