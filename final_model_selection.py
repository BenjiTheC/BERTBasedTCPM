""" The final model selection that compares Gradient Boosting algorithm
    with Neural Network
"""

import os
import json
from typing import Optional
from pprint import pprint
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    ConstantKernel,
    WhiteKernel,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, Normalizer, scale, normalize
from sklearn.model_selection import (
    cross_val_predict,
    cross_validate,
    RandomizedSearchCV,
    train_test_split,
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

from tc_data import TopCoder

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

def random_serach_top_tiers():
    """ Perform random search to find the best hyper parameters."""
    tc = TopCoder()

    model_dct = {
        'BayesianRidge': BayesianRidge,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'GaussianProcessRegressor': GaussianProcessRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'KNeighborsRegressor': KNeighborsRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'SVR': SVR,
    }

    model_args_dct = {
        'BayesianRidge': {
            'fixed_args': dict(n_iter=1000),
            'tuned_args': dict(
                tol=[1e-3, 1e-4, 1e-5, 1e-6],
            ),
        },
        'DecisionTreeRegressor': {
            'fixed_args': dict(random_state=42),
            'tuned_args': dict(
                criterion=['mse', 'mae', 'friedman_mse'],
                max_depth=[None, 3, 5, 10]
            ),
        },
        'GaussianProcessRegressor': {
            'fixed_args': dict(),
            'tuned_args': dict(kernel=[
                1.0 * RBF(),
                1.0 * RationalQuadratic(),
                ConstantKernel() * (DotProduct() ** 2),
                DotProduct() * WhiteKernel()
            ]),
        },
        'GradientBoostingRegressor': {
            'fixed_args': dict(random_state=42, n_iter_no_change=5),
            'tuned_args': dict(
                loss=['ls', 'lad'],
                n_estimators=[200, 500, 1000, 1500],
                learning_rate=[0.01, 0.001, 1e-4],
                tol=[0.01, 0.001, 1e-4, 1e-5, 2e-5, 1e-6],
            ),
        },
        'KNeighborsRegressor': {
            'fixed_args': dict(n_jobs=-1),
            'tuned_args': dict(
                n_neighbors=[5, 10, 15, 20],
                weights=['uniform', 'distance'],
                algorithm=['ball_tree', 'kd_tree'],
                leaf_size=[30, 60, 100],
            ),
        },
        'RandomForestRegressor': {
            'fixed_args': dict(n_jobs=-1, verbose=1, random_state=42, bootstrap=True),
            'tuned_args': dict(
                n_estimators=[100, 200, 500, 1000],
                max_features=['auto', 'sqrt', 0.333],
                criterion=['mae', 'mse'],
            ),
        },
        'SVR': {
            'fixed_args': dict(cache_size=15000),
            'tuned_args': [
                dict(kernel=['rbf'], gamma=['scale', 'auto'], C=[1, 10, 100, 1000]),
                dict(kernel=['linear'], C=[1, 10, 100, 1000]),
                dict(kernel=['poly'], degree=[2, 3, 5], coef0=[0, 0.5, 5, 50, 100], C=[1, 10, 100, 1000]),
            ],
        },
    }

    scoring = {
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'mre': make_scorer(mre, greater_is_better=False),
    }

    rs_path = os.path.join(os.curdir, 'result', 'random_search_res')

    with open(os.path.join(os.curdir, 'result', 'simple_regression', 'top4_reg_dct.json')) as f:
        top_regs_dct = {target: list(metrics.keys()) for target, metrics in json.load(f).items() if target != 'price'}

    for target, reg_lst in top_regs_dct.items():
        print(f'{target} | Random Searching....')
        X, y = tc.build_final_dataset(target)
        Xnp, ynp = X.to_numpy(), y.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(Xnp, ynp, test_size=0.3, random_state=42)

        for reg_name in reg_lst:
            print(f'RS on {reg_name}...')

            rs_res_path = os.path.join(rs_path, f'{target}_{reg_name}_rs.json')
            if os.path.isfile(rs_res_path):
                continue

            reg = model_dct[reg_name]
            args = model_args_dct[reg_name]
            
            rs = RandomizedSearchCV(
                reg(**args['fixed_args']),
                param_distributions=args['tuned_args'],
                n_iter=6,
                scoring=scoring,
                refit='mre',
                n_jobs=-1,
                cv=10,
                random_state=42,
            )
            rs.fit(X_train, y_train)

            rs_res = {
                'regressor': reg_name,
                'best_params': rs.best_params_,
                'best_score_in_rs': rs.best_score_,
            }

            with open(rs_res_path, 'w') as f:
                json.dump(rs_res, f, indent=4)

def kfold_predict_validate_gradient_boosting(X: pd.DataFrame, y: pd.Series, cv=10, loss='ls', tol=0.01, n_iter_no_change=5):
    """ Perform K-Fold validation and prediction for gradient boosting."""
    if not all(X.index == y.index):
        raise ValueError('Index of X and y are not equal!')

    kfold = KFold(n_splits=cv)
    cha_id_arr = np.array(X.index)

    Xnp, ynp = X.to_numpy(), y.to_numpy()

    pred_sr_lst = []
    cv_feature_importance = []
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

        cv_feature_importance.append(gbreg.feature_importances_)

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

    return y_pred, pd.DataFrame.from_records(cv_eval_res), overall_score, pd.DataFrame(cv_feature_importance, columns=X.columns)

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

def train_gb_for_production(X: pd.DataFrame, y: pd.Series, target: str, loss='ls', tol=0.01, n_iter_no_change=5):
    """ Train Gradient Boosting model for production."""
    if not all(X.index == y.index):
        raise ValueError('The indices of X and y are not equal.')

    Xnp, ynp = X.to_numpy(), y.to_numpy()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (f'{target}_gb_reg', GradientBoostingRegressor(
            n_estimators=2000,
            validation_fraction=0.2,
            random_state=42,
            verbose=1,
            loss=loss,
            tol=tol,
            n_iter_no_change=n_iter_no_change,
        ))
    ])
    pipeline.fit(Xnp, ynp)

    estimator_path = os.path.join(os.curdir, 'result', 'final_models', f'{target}_estimator.joblib')
    joblib.dump(pipeline, estimator_path)

    return pipeline

if __name__ == "__main__":
    random_serach_top_tiers()
