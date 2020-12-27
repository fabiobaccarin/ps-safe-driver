"""
Research and exploration
"""

import json
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style='whitegrid')
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.impute import MissingIndicator
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path


p = Path(__file__).parents[1]

# Setup to load modules in `src`
import sys
sys.path.append(str(p))
from src.preprocessor import Preprocessor
from src.ranker import Ranker
from src.logger import logger


# Load data
logger.info('Load data')
df = pd.read_pickle(p.joinpath('data', 'interim', 'research.pkl'))

# Prepare X
logger.info('Prepare X')
X = df.drop(labels='target', axis=1)
y = df['target'].copy()
X = Preprocessor().fit_transform(X, y).drop(labels='ps_car_10_cat', axis=1) # constant feature
del df

# Persist processed data for research
logger.info('Persist processed data for research')
X.join(y).to_pickle(p.joinpath('data', 'processed', 'research.pkl'))

# Profile on X
logger.info('Profile on X_research')
ProfileReport(
    df=X,
    title='X_research',
    config_file=p.joinpath('src', 'ProfileConf.yml')
).to_file(p.joinpath('reports', 'profiles', 'X_research.html'))

# Create ranking
logger.info('Create ranking')
ranking = Ranker().rank(X, y)

# Figure 1: feature ranking
logger.info('Figure 1: feature ranking')
fig, ax = plt.subplots(figsize=(11.7, 8.27))
ax = sns.scatterplot(
    x=ranking['significance'],
    y=ranking['assoc'],
    s=100,
)
ax.set(
    title='Figure 1: Volcano plot for features',
    xlabel='Significance (-log10(p-value))',
    ylabel='Association measure (100 scale)'
)
ax.axvline(x=0, color='black', lw=2)
ax.axhline(y=0, color='black', lw=2)
plt.tight_layout()
fig.savefig(
    fname=p.joinpath('reports', 'plots', '01FeatureRanking.png'),
    dpi=800,
    format='png'
)
plt.close(fig)

# Table 1: Feature ranking
logger.info('Table 1: feature ranking')
ranking['rank'] = ranking['assoc'].abs().rank(ascending=False)
ranking['group'] = np.ceil(ranking['rank'].div(5))
ranking.sort_values(by='rank').to_html(
    buf=p.joinpath('reports', 'tables', '01FeatureRanking.html'),
    float_format='{:.2f}'.format,
    bold_rows=False
)

# Persisting ranking
logger.info('Persist ranking')
ranking.to_pickle(p.joinpath('src', 'meta', 'ranking.pkl'))

# Correlation matrix
logger.info('Correlation matrix')
corr = X.corr()
col_order = corr.columns[AgglomerativeClustering().fit_predict(corr).argsort()]

# Figure 2: correlation matrix
logger.info('Figure 2: correlation matrix')
fig, ax = plt.subplots(figsize=(11.7, 8.27))
ax = sns.heatmap(data=corr.loc[col_order, col_order], cmap='RdBu')
ax.set(title='Figure 2: Correlation matrix of features')
plt.tight_layout()
fig.savefig(
    fname=p.joinpath('reports', 'plots', '02FeaturesCorrMatrix.png'),
    dpi=800,
    format='png'
)
plt.close(fig=fig)

# Figure 3: correlation matrix - 75 percent only
logger.info('Figure 3: correlation matrix - 75 percent only')
fig, ax = plt.subplots(figsize=(11.7, 8.27))
ax = sns.heatmap(data=corr.loc[col_order, col_order].gt(0.75), cmap='Greys', cbar=False)
ax.set(title='Figure 3: Correlation matrix of features - only correlations above 75 percent')
plt.tight_layout()
fig.savefig(
    fname=p.joinpath('reports', 'plots', '03FeaturesCorr75Matrix.png'),
    dpi=800,
    format='png'
)
plt.close(fig=fig)