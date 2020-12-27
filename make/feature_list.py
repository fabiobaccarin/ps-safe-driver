"""
Feature list of model. See `benchmark.py` for details
"""

import json
import typing as t
import pandas as pd
from pathlib import Path


def remove_suffix(name: str, suffix: t.Union[str, t.Tuple[str]]) -> str:
    return name[:-len(suffix)-1] if name.endswith(suffix) else name


p = Path(__file__).parents[1]

rnk = pd.read_pickle(p.joinpath('src', 'meta', 'ranking.pkl'))

features = [remove_suffix(name, ('cat', 'bin', 'na_bin')) for name in rnk[rnk['group'] == 1].index]
json.dump(obj=features, fp=open(file=p.joinpath('src', 'meta', 'vars.json'), mode='w'), indent=4)