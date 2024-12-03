from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, cols_drop=None, cols_get_float=None):
        self.cols_drop = cols_drop if cols_drop else []
        self.cols_get_float = cols_get_float if cols_get_float else []

    @staticmethod
    def safe_float_conversion(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if self.cols_drop:
            df = df.drop(self.cols_drop, axis=1, errors="ignore")

        if self.cols_get_float:
            for col in self.cols_get_float:
                df[col] = df[col].apply(lambda x: self.safe_float_conversion(x.split(' ')[0]) if isinstance(x, str) else x)
        return df
