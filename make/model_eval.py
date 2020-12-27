"""
Evaluates chosen models for comparison
"""

import json
import joblib
import pandas as pd
import numpy as np
import seaborn as sns; sns.set(style='whitegrid')
from matplotlib import pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from pathlib import Path


p = Path(__file__).parents[1]

# Setup to load modules in `src`
import sys
sys.path.append(str(p))
from src.logger import logger


CROSS_VAL_OPTS = {
    'scoring': 'average_precision',
    'cv': RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0),
    'n_jobs': 3,
    'return_train_score': False,
    'return_estimator': False,
}

# Load data
logger.info('Load data')
X = pd.read_pickle(p.joinpath('data', 'interim', 'X.pkl'))
y = pd.read_pickle(p.joinpath('data', 'interim', 'y.pkl'))

# Load models
logger.info('Load models')
MODELS = {model.stem: joblib.load(model) for model in p.joinpath('models').iterdir()}    

# Evaluate
bm = pd.DataFrame()
for name, model in MODELS.items():
    logger.info(f'Run {name}')
    cv = dict(cross_validate(estimator=model, X=X, y=y, **CROSS_VAL_OPTS))
    bm = pd.concat([bm, pd.DataFrame(cv).assign(model=name)])
tbl = bm.groupby('model')

# Table 3: Model performance comparison
logger.info('Table 3: Model performance comparison')
(tbl.mean().add_suffix('_mean')
    .join(tbl.std().div(np.sqrt(CROSS_VAL_OPTS['cv'].get_n_splits())).add_suffix('_stderr'))
    .assign(
        fit_time_mean=lambda df: df['fit_time_mean'].mul(1000),
        score_time_mean=lambda df: df['score_time_mean'].mul(1000),
        test_score_mean=lambda df: df['test_score_mean'].mul(100),
        fit_time_stderr=lambda df: df['fit_time_stderr'].mul(1000),
        score_time_stderr=lambda df: df['score_time_stderr'].mul(1000),
        test_score_stderr=lambda df: df['test_score_stderr'].mul(100),
        cer=lambda df: df.filter(['fit_time_mean', 'score_time_mean']).sum(axis=1).div(df['test_score_mean'])
    )
    .sort_values('cer')
    .rename(columns={
        'fit_time_mean': 'Mean fit time (ms)',
        'score_time_mean': 'Mean score time (ms)',
        'test_score_mean': 'Mean test average precision (%)',
        'fit_time_stderr': 'Fit time standard error (ms)',
        'score_time_stderr': 'Score time standard error (ms)',
        'test_score_stderr': 'Test average precision standard error (%)',
        'cer': 'Cost-effectiveness ratio'
    })
    .to_html(
        buf=p.joinpath('reports', 'tables', '03ModelPerformanceComparison.html'),
        float_format='{:.2f}'.format,
        bold_rows=False
    ))