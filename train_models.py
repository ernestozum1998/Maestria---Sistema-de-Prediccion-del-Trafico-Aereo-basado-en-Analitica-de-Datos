# train_models.py
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import optuna
from prophet import Prophet

# ==============================
# 0) RUTAS Y CARPETAS
# ==============================
ARTIF_DIR = "artifacts"
PLOTS_DIR = "plots"
os.makedirs(ARTIF_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==============================
# 1) CARGA DE DATOS (MySQL)
# ==============================
user = "root"
password = "root"
host = "localhost"
port = "3306"
database = "atc_flight_data"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
df = pd.read_sql("""
SELECT anio, mes, dia, hora, total_vuelos
FROM vuelos_por_hora
ORDER BY anio, mes, dia, hora
""", engine)
engine.dispose()

df = df.rename(columns={"anio":"year","mes":"month","dia":"day","hora":"hour"})
df['fecha_hora'] = pd.to_datetime(df[['year','month','day','hour']])
df = df[['fecha_hora','total_vuelos']].set_index('fecha_hora').sort_index()

# Outlier handling (IQR)
Q1, Q3 = df['total_vuelos'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df['total_vuelos'] = np.clip(df['total_vuelos'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# ==============================
# 2) PROPHET ENTRENADO SOLO CON 2023
# ==============================
df_2023 = df[df.index.year == 2023]
df_prophet = df_2023.reset_index()[['fecha_hora', 'total_vuelos']].rename(columns={'fecha_hora': 'ds', 'total_vuelos': 'y'})

prophet = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
# (opcional) añadir feriados país: prophet.add_country_holidays(country_name='MX')
prophet.fit(df_prophet)

# Guardar Prophet
with open(os.path.join(ARTIF_DIR, "prophet.pkl"), "wb") as f:
    pickle.dump(prophet, f)

# Predicción Prophet (2023 + 2024 bisiesto)
future = prophet.make_future_dataframe(periods=24*366, freq='H')
forecast = prophet.predict(future)
df_prophet_pred = forecast[['ds', 'yhat']].set_index('ds')

df['prophet_pred'] = df_prophet_pred['yhat'].reindex(df.index)
df['prophet_pred'] = df['prophet_pred'].interpolate()

# ===== Componentes de Prophet como features (si existen) =====
comp_cols = [c for c in ['trend', 'weekly', 'yearly', 'holidays'] if c in forecast.columns]
if comp_cols:
    comp_df = forecast.set_index('ds')[comp_cols].reindex(df.index).interpolate()
    comp_df = comp_df.add_prefix('prop_')  # prop_trend, prop_weekly, ...
    df = pd.concat([df, comp_df], axis=1)

# ==============================
# 3) FEATURE ENGINEERING (temporales + residuales)
# ==============================
festivos = pd.to_datetime([
    "2023-01-01","2023-04-06","2023-12-25",
    "2024-01-01","2024-04-06","2024-12-25"
])

df['residual'] = df['total_vuelos'] - df['prophet_pred']

# temporales
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
df['is_holiday'] = df.index.normalize().isin(festivos).astype(int)

# codificación cíclica
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24.0)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24.0)
df['dow_sin']  = np.sin(2*np.pi*df['day_of_week']/7.0)
df['dow_cos']  = np.cos(2*np.pi*df['day_of_week']/7.0)

# lags y rolling sobre residuales (shift(1) para evitar fuga)
df['lag_1h']  = df['residual'].shift(1)
df['lag_6h']  = df['residual'].shift(6)
df['lag_12h'] = df['residual'].shift(12)
df['lag_24h'] = df['residual'].shift(24)
df['rolling_6h']  = df['residual'].shift(1).rolling(6).mean()
df['rolling_12h'] = df['residual'].shift(1).rolling(12).mean()
df['rolling_24h'] = df['residual'].shift(1).rolling(24).mean()

# ===== NUEVAS FEATURES (volatilidad/extremos, semanal, interacción, ciclo anual) =====
df['rolling_std_24h'] = df['residual'].shift(1).rolling(24).std()
df['rolling_std_12h'] = df['residual'].shift(1).rolling(12).std()
df['rolling_max_24h'] = df['residual'].shift(1).rolling(24).max()
df['rolling_min_24h'] = df['residual'].shift(1).rolling(24).min()
df['lag_168h']       = df['residual'].shift(24*7)
df['hour_x_weekend'] = df['hour'] * df['is_weekend']
df['doy']     = df.index.dayofyear
df['doy_sin'] = np.sin(2*np.pi*df['doy']/366.0)
df['doy_cos'] = np.cos(2*np.pi*df['doy']/366.0)

# Targets multihorizonte
df['target_6h']  = df['residual'].shift(-6)
df['target_12h'] = df['residual'].shift(-12)
df['target_24h'] = df['residual'].shift(-24)

# ===== Pesos por nivel (más peso en horas pico) =====
lvl = df['prophet_pred'].clip(lower=0)
p5, p95 = np.percentile(lvl.dropna(), [5, 95])
den = max(p95 - p5, 1e-6)
w_raw = ((lvl - p5) / den).clip(0.0, 1.0)
ALPHA = 1.0  # 0.3–2.0 según sensibilidad deseada
df['sample_weight'] = 1.0 + ALPHA * w_raw

df = df.dropna().copy()

# lista de features base (incluye nuevas + componentes prophet si existen)
features_base = [
    'hour','day_of_week','month','is_weekend','is_holiday',
    'lag_1h','lag_6h','lag_12h','lag_24h','lag_168h',
    'rolling_6h','rolling_12h','rolling_24h',
    'rolling_std_24h','rolling_std_12h','rolling_max_24h','rolling_min_24h',
    'hour_sin','hour_cos','dow_sin','dow_cos',
    'hour_x_weekend','doy_sin','doy_cos'
]
for c in ['prop_trend','prop_weekly','prop_yearly','prop_holidays']:
    if c in df.columns and c not in features_base:
        features_base.append(c)

# ==============================
# 4) UTILIDADES XGBOOST + OPTUNA (wrapper compatible)
# ==============================
class _BoosterRegressor:
    """Wrapper para usar Booster con .predict() y exponer best_iteration."""
    def __init__(self, booster, best_iteration, params_print=None):
        self.booster = booster
        self.best_iteration = best_iteration
        self.params_print = params_print or {}

    def predict(self, X):
        dm = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        iters = (0, self.best_iteration + 1) if self.best_iteration is not None else None
        return self.booster.predict(dm, iteration_range=iters)

    def get_xgb_params(self):
        return self.params_print

def entrenar_modelo_optuna(df_train, features, target, usar_gpu=True, n_trials=60, sample_weight=None):
    """
    Entrena con Optuna usando xgb.train + DMatrix (early stopping).
    Admite 'sample_weight' (ponderación por ejemplo).
    """
    X_all = df_train[features].astype(np.float32).values
    y_all = df_train[target].astype(np.float32).values

    n = len(df_train)
    split = int(n * 0.8)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_va, y_va = X_all[split:],  y_all[split:]

    if sample_weight is not None:
        w_all = sample_weight.astype(np.float32).values
        w_tr, w_va = w_all[:split], w_all[split:]
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va, weight=w_va)
    else:
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va)

    def objective(trial):
        grow_policy = trial.suggest_categorical('grow_policy', ['depthwise','lossguide'])
        params = {
            'objective':'reg:squarederror',
            'eval_metric':'rmse',
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 20.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 50.0, log=True),
            'max_bin': trial.suggest_int('max_bin', 256, 1024),
            'seed': 42, 'verbosity': 0,
            'grow_policy': grow_policy,
            'tree_method': 'hist',
            'device': 'cuda' if usar_gpu else 'cpu'
        }
        if grow_policy == 'depthwise':
            params['max_depth'] = trial.suggest_int('max_depth', 4, 10)
        else:
            params['max_depth'] = 0
            params['max_leaves'] = trial.suggest_int('max_leaves', 31, 512)

        num_round = trial.suggest_int('n_estimators', 400, 2000)
        booster = xgb.train(params, dtrain, num_boost_round=num_round,
                            evals=[(dvalid,'valid')],
                            early_stopping_rounds=200,
                            verbose_eval=False)
        return float(booster.best_score)**2  # devolver MSE

    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    final_params = {
        'objective':'reg:squarederror','eval_metric':'rmse',
        'seed':42,'verbosity':1,'grow_policy':best_params['grow_policy'],
        'learning_rate':best_params['learning_rate'],
        'subsample':best_params['subsample'],
        'colsample_bytree':best_params['colsample_bytree'],
        'colsample_bylevel':best_params['colsample_bylevel'],
        'min_child_weight':best_params['min_child_weight'],
        'gamma':best_params['gamma'],
        'reg_alpha':best_params['reg_alpha'],
        'reg_lambda':best_params['reg_lambda'],
        'max_bin':best_params['max_bin'],
        'tree_method':'hist','device':'cuda' if usar_gpu else 'cpu'
    }
    if best_params['grow_policy'] == 'depthwise':
        final_params['max_depth'] = best_params['max_depth']
    else:
        final_params['max_depth'] = 0
        final_params['max_leaves'] = best_params['max_leaves']

    # re-arma DMatrix final con pesos si aplica
    if sample_weight is not None:
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va, weight=w_va)
    else:
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va)

    booster = xgb.train(final_params, dtrain,
                        num_boost_round=best_params['n_estimators'],
                        evals=[(dvalid,'valid')],
                        early_stopping_rounds=200,
                        verbose_eval=True)

    print("\n⚙️ XGBoost params finales:", final_params)
    try:
        print("🔹 Mejor iteración:", booster.best_iteration, "| Mejor RMSE (valid):", booster.best_score)
    except Exception:
        pass

    return _BoosterRegressor(booster, booster.best_iteration, params_print=final_params)

# ==============================
# 5) ENTRENAR MODELOS +6h y +12h (con pesos)
# ==============================
df_train = df[df.index.year == 2023].copy()
w_train = df_train['sample_weight']

model_6h  = entrenar_modelo_optuna(df_train, features_base, 'target_6h',
                                   usar_gpu=True, n_trials=60, sample_weight=w_train)
model_12h = entrenar_modelo_optuna(df_train, features_base, 'target_12h',
                                   usar_gpu=True, n_trials=60, sample_weight=w_train)

# Predicciones base como features en TODO el dataset
X_all_base = df[features_base].astype(np.float32)
df['pred_6h']  = model_6h.predict(X_all_base)
df['pred_12h'] = model_12h.predict(X_all_base)

# STACKING (+24h)
features_stacked = features_base + ['pred_6h','pred_12h']
df_train = df[df.index.year == 2023].copy()
w_train = df_train['sample_weight']

model_24h_stacked = entrenar_modelo_optuna(df_train, features_stacked, 'target_24h',
                                           usar_gpu=True, n_trials=60, sample_weight=w_train)

# ==============================
# 6) GUARDAR ARTEFACTOS
# ==============================
model_6h.booster.save_model(os.path.join(ARTIF_DIR, "xgb_6h.json"))
model_12h.booster.save_model(os.path.join(ARTIF_DIR, "xgb_12h.json"))
model_24h_stacked.booster.save_model(os.path.join(ARTIF_DIR, "xgb_24h_stacked.json"))

with open(os.path.join(ARTIF_DIR, "feature_list_base.json"), "w") as f:
    json.dump(features_base, f, indent=2)
with open(os.path.join(ARTIF_DIR, "feature_list_stacked.json"), "w") as f:
    json.dump(features_stacked, f, indent=2)

meta = {
    "stacking": True,
    "models": {
        "xgb_6h":  {"path":"xgb_6h.json", "best_iteration": model_6h.best_iteration},
        "xgb_12h": {"path":"xgb_12h.json", "best_iteration": model_12h.best_iteration},
        "xgb_24h": {"path":"xgb_24h_stacked.json", "best_iteration": model_24h_stacked.best_iteration}
    }
}
with open(os.path.join(ARTIF_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ Artefactos guardados en ./artifacts")

# ==============================
# 7) EVALUACIÓN EN 2024 (+24h STACKED)
# ==============================
df_test = df[df.index.year == 2024].copy()

# Para alinear: target_24h implica usar X(t) y comparar con y(t+24)
X_test_stacked = df_test[features_stacked].astype(np.float32).iloc[:-24]
y_true_idx = df_test.index[24:]  # índices donde existe y(t+24)

pred_resid_24h = model_24h_stacked.predict(X_test_stacked)
yhat_prophet_24h = df_prophet_pred.loc[y_true_idx, 'yhat'].values
final_pred = yhat_prophet_24h + pred_resid_24h
y_true = df.loc[y_true_idx, 'total_vuelos'].values

mse_stacked = mean_squared_error(y_true, final_pred)
acc_stacked = 100 - (np.abs(y_true - final_pred) / np.maximum(y_true, 1e-6) * 100)
acc_stacked = np.clip(acc_stacked, 0, 100)

print(f"\n📊 Modelo Stacked +24h -> MSE: {mse_stacked:.2f} | Precisión: {np.mean(acc_stacked):.2f}%")

# ==============================
# 8) GRÁFICO comparativo (Primera semana 2024)
# ==============================
plot_end = y_true_idx[0] + pd.Timedelta(hours=24*7-1)
mask = (y_true_idx >= y_true_idx[0]) & (y_true_idx <= plot_end)
idx_plot = y_true_idx[mask]

plt.figure(figsize=(18,6))
plt.plot(idx_plot, df.loc[idx_plot, 'total_vuelos'], label='Real', color='black')
plt.plot(idx_plot, final_pred[:len(idx_plot)], label='Predicción Stacked +24h', linestyle='--', color='red')
plt.title("Stacking: Predicción +24h vs Real (Primera semana 2024)")
plt.xlabel("Fecha y Hora"); plt.ylabel("Total de Vuelos")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "00_real_vs_pred_primera_semana.png"), dpi=140, bbox_inches="tight")
plt.close()

# ==============================
# 9) DIAGNÓSTICOS Y GRÁFICOS (automático)
# ==============================
def _safe_name(title: str) -> str:
    return "".join([c if c.isalnum() or c in "._-" else "_" for c in title])

def _savefig(fig, title):
    path = os.path.join(PLOTS_DIR, _safe_name(title) + ".png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 {title} -> {path}")

# Series y residuos
idx_eval = pd.DatetimeIndex(y_true_idx)
y_true_s = pd.Series(y_true, index=idx_eval, name="real")
y_pred_s = pd.Series(final_pred, index=idx_eval, name="pred")
residuals = y_true_s - y_pred_s

# 1) Real vs Predicho (vista amplia)
fig = plt.figure(figsize=(18,5))
plt.plot(idx_eval, y_true_s, label="Real", color="black")
plt.plot(idx_eval, y_pred_s, label="Predicho", color="red", linestyle="--")
plt.title("Real vs Predicho (XGBoost stacked + Prophet)")
plt.xlabel("Fecha"); plt.ylabel("Total de vuelos/hora"); plt.grid(True); plt.legend()
_savefig(fig, "01_real_vs_predicho_full")

# 2) Error porcentual en el tiempo
error_pct = (np.abs(y_true_s - y_pred_s) / np.maximum(y_true_s, 1e-6)) * 100
error_pct = error_pct.clip(0, 300)
fig = plt.figure(figsize=(18,4))
plt.plot(idx_eval, error_pct, color="orange")
plt.title("Error porcentual (%) a lo largo del tiempo")
plt.xlabel("Fecha"); plt.ylabel("Error (%)"); plt.grid(True)
_savefig(fig, "02_error_porcentual_tiempo")

# 3) Histograma de residuos
fig = plt.figure(figsize=(8,5))
plt.hist(residuals.dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title("Distribución de errores (residuos)")
plt.xlabel("Error (Real - Predicho)"); plt.ylabel("Frecuencia")
_savefig(fig, "03_histograma_residuos")

# 4) Residuos vs Predicción
fig = plt.figure(figsize=(7,6))
plt.scatter(y_pred_s, residuals, alpha=0.35)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuos vs Predicción")
plt.xlabel("Valor predicho"); plt.ylabel("Error residual")
plt.grid(True, alpha=0.3)
_savefig(fig, "04_residuos_vs_prediccion")

# 5) Residuos en el tiempo
fig = plt.figure(figsize=(18,4))
plt.plot(idx_eval, residuals, color='purple', alpha=0.9)
plt.axhline(0, color='black', linestyle='--')
plt.title("Residuos en el tiempo")
plt.xlabel("Fecha"); plt.ylabel("Error (Real - Predicho)")
plt.grid(True)
_savefig(fig, "05_residuos_en_el_tiempo")

# 6) QQ-Plot
try:
    import scipy.stats as stats
    fig = plt.figure(figsize=(6,6))
    stats.probplot(residuals.dropna(), dist="norm", plot=plt)
    plt.title("QQ-Plot de residuos")
    _savefig(fig, "06_qqplot_residuos")
except Exception as e:
    print("⚠️ QQ-Plot omitido (instala scipy). Detalle:", repr(e))

# 7) Importancia de variables (XGBoost)
try:
    booster = getattr(model_24h_stacked, "booster", None)
    if booster is not None:
        fig, ax = plt.subplots(figsize=(10,7))
        xgb.plot_importance(booster, importance_type='gain', max_num_features=20, ax=ax)
        ax.set_title("Importancia de características (Gain) - Modelo +24h Stacked")
        _savefig(fig, "07_importancia_variables_gain")
except Exception as e:
    print("⚠️ plot_importance omitido:", repr(e))

# 8) SHAP summary (global)
try:
    import shap
    if isinstance(X_test_stacked, np.ndarray):
        X_shap = pd.DataFrame(X_test_stacked, columns=features_stacked)
    else:
        X_shap = X_test_stacked.copy()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(X_shap, check_additivity=False)
    shap.summary_plot(shap_values, X_shap, show=False, plot_size=(10,6))
    plt.title("SHAP Summary Plot (global) - Modelo +24h")
    _savefig(plt.gcf(), "08_shap_summary")
except Exception as e:
    print("⚠️ SHAP omitido (instala shap). Detalle:", repr(e))

# 9) Mapa de calor de correlaciones (train)
try:
    import seaborn as sns
    cols_corr = [c for c in features_stacked if c in df_train.columns] + (['target_24h'] if 'target_24h' in df_train.columns else [])
    corr = df_train[cols_corr].corr()
    fig = plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Mapa de calor de correlaciones (train, stacked + target)")
    _savefig(plt.gcf(), "10_heatmap_correlaciones")
except Exception as e:
    print("⚠️ Heatmap de correlaciones omitido:", repr(e))

# 10) Error medio por hora del día / día de la semana
try:
    df_eval = pd.DataFrame({
        "hora": idx_eval.hour,
        "dow": idx_eval.dayofweek,
        "error_abs": (y_true_s - y_pred_s).abs(),
        "error_pct": error_pct
    })
    err_hora = df_eval.groupby("hora")["error_abs"].mean()
    fig = plt.figure(figsize=(10,4))
    plt.bar(err_hora.index, err_hora.values)
    plt.title("Error absoluto medio por hora del día")
    plt.xlabel("Hora (0-23)"); plt.ylabel("Error absoluto medio")
    plt.grid(axis='y', alpha=0.4)
    _savefig(fig, "11_error_medio_por_hora")

    err_dow = df_eval.groupby("dow")["error_abs"].mean()
    fig = plt.figure(figsize=(10,4))
    plt.bar(err_dow.index, err_dow.values)
    plt.title("Error absoluto medio por día de la semana (0=Lunes)")
    plt.xlabel("Día de la semana"); plt.ylabel("Error absoluto medio")
    plt.grid(axis='y', alpha=0.4)
    _savefig(fig, "11b_error_medio_por_dow")
except Exception as e:
    print("⚠️ Error por hora/DOW omitido:", repr(e))

print("\n✅ Gráficos generados en:", os.path.abspath(PLOTS_DIR))
