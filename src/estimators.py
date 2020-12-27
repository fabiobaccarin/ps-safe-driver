"""
Estimators
"""

import typing as t
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalGrouper(BaseEstimator, TransformerMixin):
    """ Groups labels with less than 5 percent of samples in categotical features """

    def fit(self, X: np.ndarray, y: t.Optional[pd.Series] = None) -> 'CategoricalGrouper':
        self.mapping = {}
        Xt = pd.DataFrame(X).add_prefix('x')
        for c in Xt:
            counts = Xt[c].value_counts(normalize=True)
            remove = counts[counts.lt(0.05)].index.to_list()
            new = Xt[c].replace(to_replace=remove, value=np.nan)
            # if NaN's are less than 5 per cent of samples, assign them to the most frequent label
            self.mapping[c] = {
                'remove': remove,
                'fill_value': -1 if new.isna().sum() > 0.05 * len(new)
                    else new.value_counts().idxmax()
            }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (
            pd.DataFrame(X).add_prefix('x')
            .apply(func=lambda s: s.replace(
                to_replace=self.mapping[s.name]['remove'],
                value=self.mapping[s.name]['fill_value']
            ))
            .values
        )


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """ Encodes categorical features by the event rate in each label """

    def fit(self, X: np.ndarray, y: pd.Series) -> 'CategoricalEncoder':
        Xt = pd.DataFrame(X).add_prefix('x')
        df = Xt.join(other=y)
        self.mapping = {v: df.groupby(v)[y.name].mean() for v in Xt}
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (
            pd.DataFrame(X).add_prefix('x')
            .apply(func=lambda s: s.map(arg=self.mapping[s.name]))
            .values
        )