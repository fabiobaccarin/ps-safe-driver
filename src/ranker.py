import pandas as pd
import numpy as np
import typing as t
from scipy import stats as ss


class Ranker:
    """ Ranks features. Uses Spearman's correlation for numerical features and Cramer's V for
        binary features
    """

    @staticmethod
    def _cramer_v(x: pd.Series, y: pd.Series, /) -> t.Tuple[float, float]:
        cm = pd.crosstab(x, y)
        chi2, pval, _, _ = ss.chi2_contingency(cm)
        n = cm.sum().sum()
        phi2 = chi2 / n
        r, k = cm.shape
        phi2corr = max(0, phi2 - (k-1)*(r-1)/(n-1))
        rcorr = r - (r-1)**2/(n-1)
        kcorr = k - (k-1)**2/(n-1)
        return np.sqrt(phi2corr / min(kcorr-1, rcorr-1)) * 100, -np.log10(pval)

    @staticmethod
    def _spearman_r(x: pd.Series, y: pd.Series, /) -> t.Tuple[float, float]:
        r, pval = ss.spearmanr(x, y)
        return r * 100, -np.log10(pval)
    
    def rank(self, X: pd.DataFrame, y: pd.Series, /) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            data={
                v: self._cramer_v(X[v], y) if X[v].nunique() == 2 else self._spearman_r(X[v], y)
                for v in X
            },
            orient='index',
            columns=['assoc', 'significance']
        )