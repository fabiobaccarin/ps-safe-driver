"""
Prediction benchmarking
"""

import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from pathlib import Path


CROSS_VAL_OPTS = {
    'scoring': 'average_precision',
    'cv': RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0),
    'n_jobs': 3,
    'return_train_score': False,
    'return_estimator': False,
}

lr = LogisticRegression(
    penalty='none',
    random_state=0,
    solver='saga',
    max_iter=1e4
)

p = Path(__file__).parents[1]

# Setup to load modules in `src`
import sys
sys.path.append(str(p))
from src.logger import logger


# Load processed data
logger.info('Load processed data')
df = pd.read_pickle(p.joinpath('data', 'processed', 'research.pkl'))
X = df.drop(labels='target', axis=1)
y = df['target'].copy()

# Load feature ranking
logger.info('Load feature ranking')
rnk = pd.read_pickle(p.joinpath('src', 'meta', 'ranking.pkl'))

# Get groups of features
logger.info('Get groups of features')
groups = [
    rnk[rnk['group'] == i].index.to_list()
    for i in range(1, int(rnk['group'].max() + 1))
]

# Benchmarking
bm = pd.DataFrame()
for g in range(len(groups)):
    features = list(itertools.chain.from_iterable(groups[:g+1]))
    logger.info(f"Benchmarking - group {' + '.join([str(i+1) for i in range(g+1)])} = {len(features)} features")
    label = f'benchmark{g+1:02d} ({len(features):02d} features)'
    cv = dict(cross_validate(estimator=lr, X=X.filter(features), y=y, **CROSS_VAL_OPTS))
    bm = pd.concat([bm, pd.DataFrame(cv).assign(experiment=label)])
tbl = bm.groupby('experiment')

# Table 2: Benchmarks
logger.info('Table 2: Benchmarks')
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
        buf=p.joinpath('reports', 'tables', '02Benchmarks.html'),
        float_format='{:.2f}'.format,
        bold_rows=False
    ))