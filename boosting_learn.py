""" Use Boosting algorithm for average_score, num_of_reg, sub_reg_ratio prediction."""
import os
import json
from pprint import pprint
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

from smogn import smoter
from dotenv import load_dotenv

from tc_data import TopCoder

load_dotenv()
TC = TopCoder()

def build_learning_dataset(contain_docvec=False, normalize=False):
    """ Build learning dataset for prediction of 
        - avg_score
        - number_of_registration
        - sub_reg_ratio

        I assume that these target data are regressionally imbalanced, thus we should resample it before learning.
        The threshold are set as followed:
        - avg_score: 90
        - number_of_registration: 30
        - sub_reg_ratio: 0.25

        :param contain_docvec: Boolean: Whether include document vector in the feature. Default as False
        :param normalize: Boolean: Whether to normalzie the X data.
    """
    target_threshold = {
        'avg_score': {'threshold': 90, 'majority': 'up'},
        'number_of_registration': {'threshold': 30, 'majority': 'down'},
        'sub_reg_ratio': {'threshold': 0.25, 'majority': 'down'},
    }
    test_size = 954 # len(feature_df) * 0.2 ~= 953.8, use 20% of the data for testing
    storage_path = os.path.join(os.curdir, 'result', 'boosting_learn', 'learning_data')

    cha_info = TC.get_filtered_challenge_info()

    feature_df = TC\
        .get_meta_data_features(encoded_tech=True, softmax_tech=True, return_df=True)\
        .join(cha_info.reindex(['total_prize'], axis=1))
    if contain_docvec:
        feature_df = feature_df.join(pd.read_json(os.path.join(os.curdir, 'data', 'new_docvec.json'), orient='index'))
    target_df = cha_info.reindex(list(target_threshold.keys()), axis=1)
    if not (target_df.index == feature_df.index).all():
        raise ValueError('Check index of target_df and feature_df, it\'s not equal.')

    for col, threshold in target_threshold.items():
        print(f'Building dataset for {col}')
        target = target_df[col]
        test_data_fn = os.path.join(storage_path, f'{col}_test_dv{int(contain_docvec)}_norm{int(normalize)}_extreme.json')
        train_data_original_fn = os.path.join(storage_path, f'{col}_train_original_dv{int(contain_docvec)}_norm{int(normalize)}_extreme.json')
        train_data_resample_fn = os.path.join(storage_path, f'{col}_train_resample_dv{int(contain_docvec)}_norm{int(normalize)}_extreme.json')

        # Manually proportionate the test set contain 33% of minority data.
        if threshold['majority'] == 'up':
            test_index = pd.concat([
                target[target >= threshold['threshold']].sample(n=int(test_size / 3 * 2)),
                target[target < threshold['threshold']].sample(n=int(test_size / 3)),
            ]).index
        else:
            test_index = pd.concat([
                target[target <= threshold['threshold']].sample(n=int(test_size / 3 * 2)),
                target[target > threshold['threshold']].sample(n=int(test_size / 3)),
            ]).index

        X_train = feature_df.loc[~feature_df.index.isin(test_index)].sort_index()
        X_test = feature_df.loc[feature_df.index.isin(test_index)].sort_index()
        y_train = target[~target.index.isin(test_index)].sort_index()
        y_test = target[target.index.isin(test_index)].sort_index()
        if not ((X_train.index == y_train.index).all() and (X_test.index == y_test.index).all()):
            raise ValueError('Check X, y test index, they are not equal.')

        # From now on it's pure numpy till storage ;-)
        y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

        scaler = StandardScaler().fit(X_train.to_numpy())
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
        print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

        if normalize:
            normalizer = Normalizer().fit(X_train)
            X_train, X_test = normalizer.transform(X_train), normalizer.transform(X_test)

        test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
        test_data_df = pd.DataFrame(test_data)
        test_data_df.columns = [*[f'x{i}' for i in range(X_test.shape[1])], 'y']
        test_data_df.to_json(test_data_fn, orient='index')
        print(f'Test data DataFrame shape: {test_data_df.shape}')

        train_data_original = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        train_data_original_df = pd.DataFrame(train_data_original)
        train_data_original_df.columns = [*[f'x{i}' for i in range(X_test.shape[1])], 'y']
        train_data_original_df.to_json(train_data_original_fn, orient='index')
        print(f'Training data original shape: {train_data_original_df.shape}')

        attempt = 0
        while True:
            print(f'Attempt #{attempt}...')
            try:
                train_data_resample_df = smoter(data=train_data_original_df, y='y', samp_method='extreme').reset_index(drop=True) # just use the default setting for SMOGN
            except ValueError as e:
                print(f'Encounter error: "{e}", rerun the SMOGN...')
                continue
            else:
                train_data_resample_df.to_json(train_data_resample_fn, orient='index')
                print(f'Training data resample shape: {train_data_resample_df.shape}\nData stored\n\n')
                break

if __name__ == "__main__":
    for dv in False, True:
        for norm in False, True:
            print(f'========== Hyper param: dv={dv} norm={norm} ==========')
            build_learning_dataset(contain_docvec=dv, normalize=norm)
