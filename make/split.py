"""
Splits raw dataset

Strategy:
    5% - feature research and exploration = 30k
    95% - development = 570k
"""

import pandas as pd
from pathlib import Path
from io import StringIO
from sklearn.model_selection import train_test_split

p = Path(__file__).parents[1]

df = pd.read_csv(
    filepath_or_buffer=p.joinpath('data', 'raw', 'train.csv'),
    na_values=-1
).drop(labels='id', axis=1)

buf = StringIO()
df.info(buf=buf)
with open(file=p.joinpath('reports', 'DatasetInfo.txt'), mode='w', encoding='utf-8') as f:
    f.write(buf.getvalue())

research, dev = train_test_split(df, test_size=0.95, random_state=0, stratify=df['target'])

research.to_pickle(p.joinpath('data', 'interim', 'research.pkl'))
dev.to_pickle(p.joinpath('data', 'interim', 'dev.pkl'))