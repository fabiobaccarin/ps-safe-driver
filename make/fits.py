"""
Performs grid search cross-validations for various models
"""

import joblib
import json
import typing as t
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from pathlib import Path


p = Path(__file__).parents[1]

# Setup for loading modules in `src`
import sys
sys.path.append(str(p))
from src.logger import logger
from src.models import MODELS
from src.preprocessor import Preprocessor


Estimator = t.TypeVar('Estimator')

skf = StratifiedKFold(n_splits=5, random_state=0)


def make_pipeline(model) -> Pipeline:
    return Pipeline([('preprocessor', Preprocessor()), ('clf', model)])


def fit(name: str,
        model: Estimator,
        params: t.Mapping[str, float],
        filename: Path,
        X: pd.DataFrame,
        y: pd.Series, /) -> None:
    logger.info(f'Run {name}')
    m = (
        make_pipeline(model) if params is None
        else GridSearchCV(
            estimator=make_pipeline(model),
            param_grid=params,
            scoring='average_precision',
            n_jobs=3,
            cv=skf
        )
    ).fit(X, y)
    joblib.dump(
        value=m if params is None else m.best_estimator_,
        filename=filename
    )


# Load data
logger.info('Load data')
X = pd.read_pickle(p.joinpath('data', 'interim', 'X.pkl'))
y = pd.read_pickle(p.joinpath('data', 'interim', 'y.pkl'))

# Fitting
for name, model, params in MODELS:
    f = p.joinpath('models', f'{name}.pkl')
    if f.is_file():
        logger.info(f'{name} on disk. Skip')
        continue
    fit(name, model, params, f, X, y)