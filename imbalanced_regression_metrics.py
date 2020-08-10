"""
Implement precision and recall for regression according to the paper:

https://www.dcc.fc.up.pt/~ltorgo/Papers/tr09.pdf
Torgo, Luís & Ribeiro, Rita. (2009). Precision and Recall for Regression.
Lecture Notes in Computer Science. 5808. 332-346. 10.1007/978-3-642-04747-3_26. 
"""
import os
from typing import Union, Tuple

import numpy as np

class PrecisionRecallFscoreForRegression:
    """ Class for precision and recall for regression problem
        implemented according to the paper https://www.dcc.fc.up.pt/~ltorgo/Papers/tr09.pdf

        citation format:
        Torgo, Luís & Ribeiro, Rita. (2009). Precision and Recall for Regression. Lecture Notes in Computer Science. 5808. 332-346. 10.1007/978-3-642-04747-3_26.

        __init__ takes the domain specific parameters for the calculation of precision and recall.
    """

    @classmethod
    def sigmoid_base(cls, exp_pow):
        """ A sigmoid base function that generate a sigmoid output from given exp_pow."""
        return 1 / (1 + np.exp(-1 * exp_pow))

    @classmethod
    def compute_s(cls, c, decay, delta, low=False):
        """ Comput s, shape of sigmoid, for relevance function."""
        coeff = -1 if low else 1
        return coeff * np.log((delta ** -1) - 1) / np.abs(c * decay) 

    @classmethod
    def compute_loss(cls, y_true, y_pred):
        """ Loss of prediction. It's a redundant but necessary wrapping for completeness of implementation"""
        return np.abs(y_true - y_pred)

    def __init__(
        self, tE: float,
        tL: float,
        c: Union[int, float, Tuple[Union[int, float], Union[int, float]]],
        extreme: str,
        decay=0.5,
        delta=1e-4,
        k=8,
        use_smoother_alpha: bool = True,
        beta: float = 0.5
        ):
        """ `decay` is the k in paper's Equation (5)."""
        if extreme not in ('low', 'both', 'high'):
            raise ValueError(f'`extreme` should be one of ("low", "both", "high"), receive {extreme}')

        if isinstance(c, (int, float)) and extreme == 'both':
            raise ValueError(f'`c` should be a two-element tuple of floats when `extreme` equal to "both", got a number: {c}')
        if isinstance(c, tuple) and extreme in ('low', 'high'):
            raise ValueError(f'`c` should be a number when `extreme` is "low" or "high", got a tuple: {c}')
        if isinstance(c, tuple) and c[0] > c[1]:
            raise ValueError(f'c[0] should be less or equal to c[1]!, received `c`: {c}')

        if beta < 0 or beta > 1:
            raise ValueError(f'`beta` should be in the interval [0, 1], received `beta`: {beta}')

        self.tE = tE
        self.tL = tL
        self.c = c
        self.k = k # this k is for the computation of smooth-alpha
        self.use_smoother_alpha = use_smoother_alpha
        self.beta = beta

        if extreme == 'both':
            self.s = (
                self.compute_s(c[0], decay, delta, low=True),
                self.compute_s(c[1], decay, delta)
            )
        else:
            self.s = self.compute_s(c, decay, delta, low=(extreme == 'low'))

    def indicator(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ Indicator function I where return `0`/`1` for `int(loss <= tL)`"""
        return (self.compute_loss(y_true, y_pred) <= self.tL).astype(int)

    def alpha(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ Accuracy of prediction, regression version.
            A classification version is alpha(y_true, y_pred) = int(LOSS(y_true, y_pred) == 0)
            An alternative of smoother alpha is preferred.
        """
        return self.indicator(y_true, y_pred)

    def smoother_alpha(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ Smoother version of alpha propolsed by the author of the paper.
            Note that the `k` here is different from the `k` in the phi function!
        """
        return self.indicator(y_true, y_pred) * (1 - np.exp(-1 * self.k * ((self.compute_loss(y_true, y_pred) - self.tL) / self.tL) ** 2))

    def phi(self, y: np.ndarray):
        """ The relevance function of y.
            Given an integer or numpy array y, return the relevance of y, i.e phi(y)
            where phi(y) is essentially a sigmoid-like function
        """
        if isinstance(self.c, (int, float)):
            return self.sigmoid_base(self.s * (y - self.c))

        elif isinstance(self.c, tuple):
            output_arr = np.empty(y.shape) * np.nan # a new array for population of the phi

            c_low, c_high = self.c
            s_low, s_high = self.s

            # obtain the indices of values in y that are on the left/right of symmetry axis
            symmetry_point = (c_low + c_high) / 2
            y_low = np.where(y <= symmetry_point)
            y_high = np.where(y > symmetry_point)

            output_arr[y_low] = self.sigmoid_base(s_low * (y[y_low] - c_low))
            output_arr[y_high] = self.sigmoid_base(s_high * (y[y_high] - c_high))
            return output_arr

    def precision(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ Precision metric for regression problem"""
        alpha_func = self.smoother_alpha if self.use_smoother_alpha else self.alpha

        alpha = alpha_func(y_true, y_pred)
        phi_y_pred = self.phi(y_pred)

        numerator = (alpha * phi_y_pred)[np.where(phi_y_pred >= self.tE)].sum()
        denominator = phi_y_pred[np.where(phi_y_pred >= self.tE)].sum()

        return numerator / denominator

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ Recall metric for regression problem"""
        alpha_func = self.smoother_alpha if self.use_smoother_alpha else self.alpha

        alpha = alpha_func(y_true, y_pred)
        phi_y_ture = self.phi(y_true)

        numerator = (alpha * phi_y_ture)[np.where(phi_y_ture >= self.tE)].sum()
        denominator = phi_y_ture[np.where(phi_y_ture >= self.tE)].sum()

        return numerator / denominator

    def fscore(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ F-measure that aggregated precision and recall."""
        precision = self.precision(y_true, y_pred)
        recall = self.precision(y_true, y_pred)
        beta_square = self.beta ** 2

        return (beta_square + 1) * precision * recall / (beta_square * precision + recall)
