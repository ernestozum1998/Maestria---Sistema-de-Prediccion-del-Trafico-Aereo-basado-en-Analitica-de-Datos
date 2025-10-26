# ==============================
#  TRAIN MODELS - LSTM STACKED (Prophet + Optuna + GPU) - opción A con fallbacks
# ==============================

import os, json, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from prophet import Prophet
import optuna

# ==============================
# IMPORTS DEL PIPELINE (excepto las 2 funciones que faltan)
# ==============================
from feature_pipeline import (
    add_time_features,
    add_residuals,
    build_targets,
    base_feature_list
)

# ==============================
# FALLBACK 1: ATTACH PROPHET COMPONENTS
# ==============================
def attach_prophet_components_if_available(df: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Une componentes de Prophet (trend, weekly, yearly, holidays si existe) a df por índice temporal.
    Si alguna componente no existe en forecast, se crea la columna en 0 para mantener las features estables.
    """
    df = df.copy()
    available = set(forecast.columns)

    mapping = {
        "prop_trend": "trend",
        "prop_weekly": "weekly",
        "prop_yearly": "yearly",
        "prop_holidays": "holidays",  # puede no existir
    }

    comp = pd.DataFrame(index=pd.to_datetime(forecast["ds"]))
    for new_col, src in mapping.items():
        if src in available:
            comp[new_col] = forecast[src].values
        else:
            comp[new_col] = 0.0

    comp = comp.set_index(pd.to_datetime(forecast["ds"]))
    comp = comp.reindex(df.index).fillna(0.0)
    df = pd.concat([df, comp[list(mapping.keys())]], axis=1)
    return df

# ==============================
# FALLBACK 2: MAKE SAMPLE WEIGHTS
# ==============================
def make_sample_weights(df: pd.DataFrame,
                        peak_hours=(6,7,8,9,16,17,18,19,20),
                        w_base=1.0, w_peak=2.0, w_weekend=1.3, w_holiday=1.5,
                        normalize=True) -> pd.Series:
    """
    Genera pesos por muestra para ponderar la pérdida:
    - Horas pico: mayor peso (penaliza más sus errores)
    - Fin de semana y festivos: peso algo mayor
    Requiere (o crea) columnas: 'hour', 'is_weekend', 'is_holiday'.
    """
    df = df.copy()

    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df.index.dayofweek.isin([5,6]).astype(int)
    if 'is_holiday' not in df.columns:
        # si no tienes festivos integrados en add_time_features, los tratamos como 0
        df['is_holiday'] = 0

    w = np.full(len(df), w_base, dtype=np.float32)
    is_peak = df['hour'].isin(peak_hours).values
    w[is_peak] *= w_peak
    w[df['is_weekend'].values == 1] *= w_weekend
    w[df['is_holiday'].values == 1] *= w_holiday

    if normalize and w.mean() > 0:
        w = w / w.mean()

    return pd.Series(w, index=df.index, name='sample_weight')

# ==============================
# RUTAS Y DIRECTORIOS
# ==============================
ARTIF_DIR = "artifacts"
PLOTS_DIR = "plots"
os.makedirs(ARTIF_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==============================
# 1) CARGA DE DATOS (MySQL)
# ==============================
user="root"; password="root"; host="localhost"; port="3306"; database="atc_flight_data"
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
Q1, Q3 = df['total_vuelos'].quantile([0.25, 0.75]); IQR = Q3 - Q1
df['total_vuelos'] = np.clip(df['total_vuelos'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# ==============================
# 2) PROPHET ENTRENADO SOLO CON 2023
# ==============================
df_2023 = df[df.index.year == 2023]
prophet = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
prophet.fit(df_2023.reset_index().rename(columns={'fecha_hora':'ds','total_vuelos':'y'})[['ds','y']])

# Guardar Prophet
with open(os.path.join(ARTIF_DIR, "prophet.pkl"), "wb") as f:
    pickle.dump(prophet, f)

# Predicción Prophet completa (2023 + 2024 bisiesto)
future = prophet.make_future_dataframe(periods=24*366, freq='H')
forecast = prophet.predict(future)
df_prophet_pred = forecast[['ds','yhat','trend','weekly','yearly']].set_index('ds')

# ==============================
# 3) FEATURE ENGINEERING (pipeline + fallbacks)
# ==============================
df['prophet_pred'] = df_prophet_pred['yhat'].reindex(df.index).interpolate()
df = add_time_features(df)                           # hour, day_of_week, month, weekend, holiday, sin/cos, etc.
df = add_residuals(df)                               # residual = total_vuelos - prophet_pred
df = attach_prophet_components_if_available(df, forecast)  # agrega prop_trend, prop_weekly, prop_yearly, prop_holidays(0 si no hay)
df = build_targets(df)                               # target_6h/12h/24h (shift del residual)
df['sample_weight'] = make_sample_weights(df)        # pesos para entrenamiento
df = df.dropna().copy()

features_base = base_feature_list(df)                # usa tu lista desde el pipeline
print("🔧 Features base:", len(features_base))

# ==============================
# 4) SECUENCIAS
# ==============================
def build_sequences(X_df: pd.DataFrame, y: pd.Series, window: int):
    """
    Devuelve X (N, window, F) y y (N,1) alineados en el tiempo:
    cada fila i usa el rango [t-window, t-1] para predecir el target en t.
    """
    X_np = X_df.astype(np.float32).values
    y_np = y.astype(np.float32).values.reshape(-1, 1)
    xs, ys = [], []
    for t in range(window, len(X_df)):
        xs.append(X_np[t-window:t, :])
        ys.append(y_np[t, :])
    return np.stack(xs), np.stack(ys).squeeze(2)

class SeqDataset(Dataset):
    def __init__(self, X, y, sample_w=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if sample_w is None:
            self.w = torch.ones(len(self.y), dtype=torch.float32)
        else:
            # recorta las primeras 'window' filas
            self.w = torch.tensor(sample_w.astype(np.float32).values[-len(self.y):], dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]

# ==============================
# 5) MODELO LSTM
# ==============================
class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,1))
    def forward(self, x):
        out, _ = self.lstm(x)             # (B, T, H)
        return self.head(out[:, -1, :]).squeeze(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("🖥️ Device:", DEVICE)

# ==============================
# 6) ENTRENAMIENTO CON OPTUNA
# ==============================
def train_lstm_optuna(X_df, y_series, sample_w, n_trials=20, horizon_name='h'):
    """
    Búsqueda de hiperparámetros y entrenamiento final con early-stopping sencillo.
    Split temporal 80/20 interno.
    """
    n = len(X_df); split = int(n*0.8)
    X_tr, X_va = X_df.iloc[:split], X_df.iloc[split:]
    y_tr, y_va = y_series.iloc[:split], y_series.iloc[split:]
    w_tr, w_va = sample_w.iloc[:split], sample_w.iloc[split:]

    def objective(trial):
        window   = trial.suggest_int("window", 24, 96)
        hidden   = trial.suggest_int("hidden", 64, 256)
        layers   = trial.suggest_int("layers", 1, 3)
        dropout  = trial.suggest_float("dropout", 0.0, 0.5)
        lr       = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch    = trial.suggest_categorical("batch", [32, 64, 128])
        epochs   = 15

        Xtr, ytr = build_sequences(X_tr, y_tr, window)
        Xva, yva = build_sequences(X_va, y_va, window)

        dtr = SeqDataset(Xtr, ytr, w_tr)
        dva = SeqDataset(Xva, yva, w_va)
        tl = DataLoader(dtr, batch_size=batch, shuffle=True)
        vl = DataLoader(dva, batch_size=batch, shuffle=False)

        model = ResidualLSTM(input_dim=X_df.shape[1], hidden=hidden, layers=layers, dropout=dropout).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss(reduction='none')

        best = 1e9
        for _ in range(epochs):
            model.train()
            for xb, yb, wb in tl:
                xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
                opt.zero_grad()
                pred = model(xb)
                loss = (loss_fn(pred, yb)*wb).mean()
                loss.backward(); opt.step()

            # valid
            model.eval()
            Xv, yv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE), torch.tensor(yva, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                val = nn.MSELoss()(model(Xv), yv).item()
            best = min(best, val)
        return best

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"✅ Optuna {horizon_name}: {best}")

    # Entrenamiento final con los mejores HP
    window = best['window']; batch = best['batch']
    Xtr, ytr = build_sequences(X_tr, y_tr, window)
    Xva, yva = build_sequences(X_va, y_va, window)

    dtr = SeqDataset(Xtr, ytr, w_tr)
    tl = DataLoader(dtr, batch_size=batch, shuffle=True)

    model = ResidualLSTM(input_dim=X_df.shape[1], hidden=best['hidden'], layers=best['layers'], dropout=best['dropout']).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=best['lr'])
    loss_fn = nn.SmoothL1Loss(reduction='none')

    best_state=None; best_val=1e9; patience=8; since=0
    for ep in range(40):
        model.train()
        for xb, yb, wb in tl:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = (loss_fn(pred, yb)*wb).mean()
            loss.backward(); opt.step()

        # valid simple para escoger mejor estado
        model.eval()
        Xv, yv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE), torch.tensor(yva, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            v = nn.MSELoss()(model(Xv), yv).item()
        if v < best_val - 1e-4:
            best_val = v; since = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, window

def run_inference_seq(model, X_df, window):
    model.eval()
    X_np = X_df.astype(np.float32).values
    outs, idx = [], []
    with torch.no_grad():
        for t in range(window, len(X_df)):
            win = torch.tensor(X_np[t-window:t, :], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            outs.append(model(win).item()); idx.append(X_df.index[t])
    return pd.Series(outs, index=pd.Index(idx))

# ==============================
# 7) ENTRENAR +6h / +12h / +24h (STACKED)
# ==============================
mask_2023 = df.index.year == 2023
X_base_all = df[features_base].copy()
w_all = df['sample_weight'].copy()

# +6h
m6, w6 = train_lstm_optuna(X_base_all[mask_2023], df.loc[mask_2023, 'target_6h'], w_all[mask_2023],
                           n_trials=15, horizon_name='+6h')
torch.save(m6.state_dict(), os.path.join(ARTIF_DIR, 'lstm_6h.pt'))
df['pred_6h'] = run_inference_seq(m6, X_base_all, w6)

# +12h
m12, w12 = train_lstm_optuna(X_base_all[mask_2023], df.loc[mask_2023, 'target_12h'], w_all[mask_2023],
                             n_trials=15, horizon_name='+12h')
torch.save(m12.state_dict(), os.path.join(ARTIF_DIR, 'lstm_12h.pt'))
df['pred_12h'] = run_inference_seq(m12, X_base_all, w12)

# +24h STACKED (usa pred_6h y pred_12h como features)
features_stacked = features_base + ['pred_6h', 'pred_12h']
X_stacked_all = df[features_stacked].copy()

m24, w24 = train_lstm_optuna(X_stacked_all[mask_2023], df.loc[mask_2023, 'target_24h'], w_all[mask_2023],
                             n_trials=20, horizon_name='+24h')
torch.save(m24.state_dict(), os.path.join(ARTIF_DIR, 'lstm_24h_stacked.pt'))

# ==============================
# 8) GUARDAR METADATA Y FEATURES
# ==============================
with open(os.path.join(ARTIF_DIR, "feature_list_base.json"), "w") as f:
    json.dump(features_base, f, indent=2)
with open(os.path.join(ARTIF_DIR, "feature_list_stacked.json"), "w") as f:
    json.dump(features_stacked, f, indent=2)

meta = {
    "framework": "lstm_stacked",
    "window": {"h6": w6, "h12": w12, "h24": w24},
    "models": {
        "lstm_6h": "lstm_6h.pt",
        "lstm_12h": "lstm_12h.pt",
        "lstm_24h_stacked": "lstm_24h_stacked.pt"
    },
    "features_base": "feature_list_base.json",
    "features_stacked": "feature_list_stacked.json"
}
with open(os.path.join(ARTIF_DIR, "metadata_lstm.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Artefactos guardados en ./artifacts")

# ==============================
# 9) EVALUACIÓN EN 2024 (+24h STACKED)
# ==============================
mask_2024 = df.index.year == 2024
X_2024_st = X_stacked_all[mask_2024]
idx_true = X_2024_st.index[w24:]          # por la ventana
pred_resid_24 = run_inference_seq(m24, X_2024_st, w24).reindex(idx_true)

# reconstrucción: y_hat_final = prophet + residual_pred
yhat_prophet = df_prophet_pred['yhat'].reindex(idx_true)
final_pred = (yhat_prophet + pred_resid_24).dropna()
y_true = df['total_vuelos'].reindex(final_pred.index)

mse = mean_squared_error(y_true, final_pred)
acc = 100 - (np.abs(y_true - final_pred) / np.maximum(y_true, 1e-6) * 100)
print(f"\n📊 LSTM Stacked +24h -> MSE: {mse:.2f} | Precisión: {acc.mean():.2f}%")

# ==============================
# 10) GRÁFICO (Primera semana 2024)
# ==============================
start = final_pred.index.min()
end = start + pd.Timedelta(hours=24*7-1)
mask = (final_pred.index >= start) & (final_pred.index <= end)

plt.figure(figsize=(18,6))
plt.plot(final_pred.index[mask], y_true[mask], label="Real", color="black")
plt.plot(final_pred.index[mask], final_pred[mask], label="LSTM +24h (stacked)", linestyle="--", color="royalblue")
plt.title("LSTM Stacked +24h vs Real (Primera semana 2024)")
plt.xlabel("Fecha y Hora"); plt.ylabel("Total de Vuelos")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "lstm_24h_semana1_2024.png"), dpi=140, bbox_inches="tight")
plt.close()
print("🖼️ Plot guardado:", os.path.join(PLOTS_DIR, "lstm_24h_semana1_2024.png"))
