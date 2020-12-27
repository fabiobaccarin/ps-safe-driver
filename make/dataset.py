"""
Creates data set with selected features
"""

import json
import pandas as pd
from pathlib import Path


p = Path(__file__).parents[1]

# Setup to load modules in `src`
import sys
sys.path.append(str(p))
from src.logger import logger


# Load list of variables
logger.info('Load list of variables')
VARS = json.load(fp=open(file=p.joinpath('src', 'meta', 'vars.json'), mode='r'))

# Load data
logger.info('Load data')
df = pd.read_pickle(p.joinpath('data', 'interim', 'dev.pkl'))

# Filter variables and persist data
logger.info('Filter variables and persist data')
df.filter([col for col in df for v in VARS if v in col]).to_pickle(p.joinpath('data', 'interim', 'X.pkl'))
df['target'].copy().to_pickle(p.joinpath('data', 'interim', 'y.pkl'))