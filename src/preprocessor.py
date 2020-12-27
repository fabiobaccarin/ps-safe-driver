import typing as t
import pandas as pd
from src import estimators as e
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):
    """ Applies preprocessing of features, with a pipeline for different types of features """
    
    def __init__(self):
        self.missing_indicator = MissingIndicator()
        self.categorical_preprocessor = make_pipeline(
            e.CategoricalGrouper(),
            e.CategoricalEncoder(),
            SimpleImputer(strategy='most_frequent'),
            QuantileTransformer(output_distribution='normal', random_state=0),
            StandardScaler()
        )
        self.numerical_preprocessor = make_pipeline(
            SimpleImputer(strategy='median'),
            QuantileTransformer(output_distribution='normal', random_state=0),
            StandardScaler()
        )

    @staticmethod
    def columns_with_missing_rule(X: pd.DataFrame, /) -> t.List[str]:
        return X.columns[X.isna().sum().gt(0.05 * len(X)).values].to_list()

    @staticmethod
    def get_categorical_columns(X: pd.DataFrame, /) -> t.List[str]:
        return [c for c in X if c.endswith('cat')]

    @staticmethod
    def get_binary_columns(X: pd.DataFrame, /) -> t.List[str]:
        return [c for c in X if c.endswith('bin')]

    @staticmethod
    def get_numerical_columns(X: pd.DataFrame, /) -> t.List[str]:
        return [c for c in X if not c.endswith(('cat', 'bin'))]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, /) -> 'Preprocessor':
        self.binary_cols = self.get_binary_columns(X)
        
        self.missing_indicator_cols = self.columns_with_missing_rule(X)
        self.missing_indicator.fit(X.filter(self.missing_indicator_cols))
        
        self.categorical_cols = self.get_categorical_columns(X)
        self.categorical_preprocessor.fit(X.filter(self.categorical_cols), y)
        
        self.numerical_cols = self.get_numerical_columns(X)
        self.numerical_preprocessor.fit(X.filter(self.numerical_cols))
        return self

    def transform(self, X: pd.DataFrame, /) -> pd.DataFrame:
        mi = pd.DataFrame(
            data=self.missing_indicator.transform(X.filter(self.missing_indicator_cols)),
            columns=[
                v[:-3] + 'na_bin' if v.endswith(('cat', 'bin')) else v + '_na_bin'
                for v in self.missing_indicator_cols
            ],
            index=X.index
        )
        cf = pd.DataFrame(
            data=self.categorical_preprocessor.transform(X.filter(self.categorical_cols)),
            columns=self.categorical_cols,
            index=X.index
        )
        nf = pd.DataFrame(
            data=self.numerical_preprocessor.transform(X.filter(self.numerical_cols)),
            columns=self.numerical_cols,
            index=X.index
        )
        return (
            X.filter(self.binary_cols)
            .join(mi)
            .join(cf)
            .join(nf)
        )