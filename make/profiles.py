"""
Makes Pandas profiles of datasets
"""

import pandas as pd
from pathlib import Path
from pandas_profiling import ProfileReport


p = Path(__file__).parents[1]

for datafile in p.joinpath('data', 'interim').iterdir():
    ProfileReport(
        df=pd.read_pickle(datafile),
        config_file=p.joinpath('src', 'ProfileConf.yml'),
        title=datafile.stem
    ).to_file(p.joinpath('reports', 'profiles', f'{datafile.stem}.html'))