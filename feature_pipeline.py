# feature_pipeline.py
import numpy as np
import pandas as pd

FESTIVOS_DEFAULT = [
    "2023-01-01","2023-04-06","2023-12-25",
    "2024-01-01","2024-04-06","2024-12-25"
]

def add_time_features(df: pd.DataFrame, festivos=None) -> pd.DataFrame:
    """Agrega features temporales y cíclicas. df index = DatetimeIndex; requiere columna: total_vuelos, prophet_pred."""
    festivos = festivos or FESTIVOS_DEFAULT
    out = df.copy()
    out['hour'] = out.index.hour
    out['day_of_week'] = out.index.dayofweek
    out['month'] = out.index.month
    out['is_weekend'] = out['day_of_week'].isin([5, 6]).astype(int)
    fset = pd.to_datetime(festivos)
    out['is_holiday'] = out.index.normalize().isin(fset).astype(int)

    out['hour_sin'] = np.sin(2*np.pi*out['hour']/24)
    out['hour_cos'] = np.cos(2*np.pi*out['hour']/24)
    out['dow_sin']  = np.sin(2*np.pi*out['day_of_week']/7)
    out['dow_cos']  = np.cos(2*np.pi*out['day_of_week']/7)
    return out

def add_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Asume columnas total_vuelos y prophet_pred. Crea residual, lags, rolling."""
    out = df.copy()
    out['residual'] = out['total_vuelos'] - out['prophet_pred']
    # lags
    for lag in [1, 6, 12, 24]:
        out[f'lag_{lag}h'] = out['residual'].shift(lag)
    # rolling
    for r in [6, 12, 24]:
        out[f'rolling_{r}h'] = out['residual'].shift(1).rolling(r).mean()
    return out

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['target_6h']  = out['residual'].shift(-6)
    out['target_12h'] = out['residual'].shift(-12)
    out['target_24h'] = out['residual'].shift(-24)
    return out

def base_feature_list():
    return [
        'hour','day_of_week','month','is_weekend','is_holiday',
        'lag_1h','lag_6h','lag_12h','lag_24h',
        'rolling_6h','rolling_12h','rolling_24h',
        'hour_sin','hour_cos','dow_sin','dow_cos'
    ]
