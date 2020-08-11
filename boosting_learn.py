""" Use Boosting algorithm for average_score, num_of_reg, sub_reg_ratio prediction."""
import os
import json
from pprint import pprint
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor

from smogn import smoter
from dotenv import load_dotenv

from tc_data import TopCoder
from imbalanced_regression_metrics import PrecisionRecallFscoreForRegression

load_dotenv()

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
    TC = TopCoder()
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

class EnsembleTrainer:
    """ Wrapper class that takes in the algo, dataset, param_grid, training target to select:
        [best dataset] + [best_params] with best score and record the test score.
    """
    res_path = os.path.join(os.curdir, 'result', 'boosting_learn', 'model_selection')
    dataset_path = os.path.join(os.curdir, 'result', 'boosting_learn', 'learning_data')
    dataset_param_grid = {'dv': (0, 1), 'norm': (0, 1), 'strategy': ('balanced', 'extreme')}

    @classmethod
    def read_dataset(cls, target, ds_type, dv, norm, strategy):
        """ Read a dataset from boosting learn data path"""
        file_name = f'{target}_{ds_type}_dv{dv}_norm{norm}_{strategy}.json'
        dataset = pd.read_json(os.path.join(cls.dataset_path, file_name), orient='index')

        y = dataset.pop('y').to_numpy()
        X = dataset.to_numpy()

        return X, y

    def __init__(
        self,
        regressor,
        param_grid,
        target,
        metric_args,
        cv=5
        ):

        if target not in ('avg_score', 'number_of_registration', 'sub_reg_ratio'):
            raise ValueError(f'target should be one of ("avg_score", "number_of_registration", "sub_reg_ratio"), received {target}')

        self.regressor = regressor
        self.model_param_grid = param_grid
        self.target = target
        self.target_metrics = PrecisionRecallFscoreForRegression(**metric_args)
        self.cv = cv

    def gridsearch_one_dataset(self, X_train, y_train, X_test, y_test):
        """ Perform Grid Search CV with one dataset."""
        scoring = {
            'precision': make_scorer(self.target_metrics.precision),
            'recall': make_scorer(self.target_metrics.recall),
        }
        gs = GridSearchCV(
            self.regressor(),
            param_grid=self.model_param_grid,
            scoring=scoring,
            refit='precision',
            cv=self.cv,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)

        y_test_pred = gs.predict(X_test)
        return {
            'regressor': self.regressor.__name__,
            'best_params': gs.best_params_,
            'best_score_in_gs': gs.best_score_,
            'train_scrore_from_gs': gs.score(X_train, y_train),
            'test_score_from_gs': gs.score(X_test, y_test),
            'manual_test_precision': self.target_metrics.precision(y_test, y_test_pred),
            'manual_test_recall': self.target_metrics.recall(y_test, y_test_pred),
            'manual_test_fscore': self.target_metrics.fscore(y_test, y_test_pred),
        }

    def gridsearch(self, verbose=0):
        """ Perform GridSearch CV over every dataset for the target"""
        regressor_name = self.regressor.__name__.lower()
        for dv in self.dataset_param_grid['dv']:
            for norm in self.dataset_param_grid['norm']:
                for strategy in self.dataset_param_grid['strategy']:
                    if verbose:
                        print(f'\tGrid seraching... | dataset info: dv={dv}, norm={norm}, stra={strategy}')

                    X_train, y_train = self.read_dataset(self.target, 'train_resample', dv, norm, strategy)
                    X_test, y_test = self.read_dataset(self.target, 'test', dv, norm, strategy)

                    gs_result = self.gridsearch_one_dataset(X_train, y_train, X_test, y_test)
                    ds_params = {'target': self.target, 'dv': dv, 'norm': norm, 'strategy': strategy}

                    if verbose:
                        print('\tGrid seraching done. Result:')
                        pprint(gs_result, indent=2)

                    with open(os.path.join(self.res_path, f'{self.target}_{regressor_name}_dv{dv}_norm{norm}_{strategy}.json'), 'w') as fwrite:
                        json.dump({'gs': gs_result, 'ds': ds_params}, fwrite, indent=4)

def gs_all_targets():
    """ Search for avg_score training."""
    target_metric_args = {
        'avg_score': dict(tE=0.6, tL=3, c=90, extreme='low', decay=0.1),
        'number_of_registration': dict(tE=0.6, tL=8, c=30, extreme='high'),
        'sub_reg_ratio': dict(tE=0.6, tL=0.07, c=0.25, extreme='high')
    }
    regressor_param_lst = [
        (
            GradientBoostingRegressor,
            dict(
                loss=['ls', 'lad'],
                n_estimators=[200, 500, 1000, 2000, 2500, 3000, 4000],
                learning_rate=[0.1, 0.01, 1e-4, 2e-5],
                max_depth=[3, 5, 8],
                random_state=[42]
                )
        ),
        (
            RandomForestRegressor,
            dict(
                n_estimators=[200, 500, 1000, 2000, 2500, 3000, 4000],
                max_features=['auto', 'sqrt', 'log2'],
                bootstrap=[True, False],
                random_state=[42]
            )
        ),
        (
            AdaBoostRegressor,
            dict(
                base_estimator__max_depth=[3, 5, 8, 10],
                n_estimators=[200, 500, 1000, 2000, 2500, 3000, 4000],
                learning_rate=[0.1, 0.01, 1e-4, 2e-5],
                loss=['linear', 'exponential'],
                random_state=[42],
                )
        )
    ]

    for target, metirc_args in target_metric_args.items():
        for regressor, param_grid in regressor_param_lst:
            print(f'\n========== Training {regressor.__name__} with {target} ==========')
            trainer = EnsembleTrainer(regressor, param_grid, target, metirc_args)
            trainer.gridsearch(verbose=1)


if __name__ == "__main__":
    gs_all_targets()
