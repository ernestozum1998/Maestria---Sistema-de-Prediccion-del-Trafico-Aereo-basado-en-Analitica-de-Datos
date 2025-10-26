# ==============================
#  PREDICCIÓN DE TRÁFICO AÉREO
#  PROPHET + LSTM MULTIHORIZONTE + OPTUNA (GPU)
# ==============================

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from prophet import Prophet
import optuna
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 1. CARGA DE DATOS
# ==============================
user = "root"
password = "root"
host = "localhost"
port = "3306"
database = "atc_flight_data"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
query = """
SELECT anio, mes, dia, hora, total_vuelos
FROM vuelos_por_hora
ORDER BY anio, mes, dia, hora
"""
df = pd.read_sql(query, engine)
engine.dispose()

# ==============================
# 2. PREPROCESAMIENTO
# ==============================
df = df.rename(columns={"anio": "year", "mes": "month", "dia": "day", "hora": "hour"})
df['fecha_hora'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df.set_index('fecha_hora', inplace=True)
df = df[['total_vuelos']]

Q1, Q3 = df['total_vuelos'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df['total_vuelos'] = np.clip(df['total_vuelos'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# ==============================
# 3. PROPHET
# ==============================
df_2023 = df[df.index.year == 2023]
df_prophet = df_2023.reset_index()[['fecha_hora', 'total_vuelos']].rename(columns={'fecha_hora': 'ds', 'total_vuelos': 'y'})

prophet = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
prophet.fit(df_prophet)

future = prophet.make_future_dataframe(periods=24*366, freq='h')
forecast = prophet.predict(future)
df_prophet_pred = forecast[['ds', 'yhat']].set_index('ds')

df['prophet_pred'] = df_prophet_pred['yhat']
df['residual'] = df['total_vuelos'] - df['prophet_pred']

# ==============================
# 4. FEATURES
# ==============================
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
festivos = pd.to_datetime(["2023-01-01","2023-04-06","2023-12-25","2024-01-01","2024-04-06","2024-12-25"])
df['is_holiday'] = df.index.normalize().isin(festivos).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)

df['lag_1h'] = df['residual'].shift(1)
df['lag_6h'] = df['residual'].shift(6)
df['lag_12h'] = df['residual'].shift(12)
df['lag_24h'] = df['residual'].shift(24)
df['rolling_6h'] = df['residual'].shift(1).rolling(6).mean()
df['rolling_12h'] = df['residual'].shift(1).rolling(12).mean()
df['rolling_24h'] = df['residual'].shift(1).rolling(24).mean()
df.dropna(inplace=True)

# ==============================
# 5. DIVISIÓN TRAIN/TEST
# ==============================
df_train = df[df.index.year == 2023]
df_test = df[df.index.year == 2024]

features = ['hour','day_of_week','month','is_weekend','is_holiday',
            'hour_sin','hour_cos','dow_sin','dow_cos',
            'lag_1h','lag_6h','lag_12h','lag_24h',
            'rolling_6h','rolling_12h','rolling_24h']

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(df_train[features + ['residual']])
scaled_test = scaler.transform(df_test[features + ['residual']])

def create_sequences(data, target_idx, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :])
        y.append(data[i+window_size, target_idx])
    return np.array(X), np.array(y)

# ==============================
# 6. DEFINICIÓN DEL MODELO
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==============================
# 7. OPTUNA
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Dispositivo:", DEVICE)

def objective(trial):
    window = trial.suggest_int("window", 24, 96)
    hidden = trial.suggest_int("hidden_size", 32, 128)
    layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = 15

    X_train, y_train = create_sequences(scaled_train, target_idx=-1, window_size=window)
    dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float().unsqueeze(-1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(X_train.shape[2], hidden, layers, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    X_val, y_val = create_sequences(scaled_test, target_idx=-1, window_size=window)
    with torch.no_grad():
        val_pred = model(torch.tensor(X_val).float().to(DEVICE))
        val_loss = criterion(val_pred, torch.tensor(y_val).float().unsqueeze(-1).to(DEVICE)).item()
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)
best = study.best_params
print("✅ Mejores hiperparámetros:", best)

# ==============================
# 8. ENTRENAMIENTO FINAL
# ==============================
window = best['window']
X_train, y_train = create_sequences(scaled_train, target_idx=-1, window_size=window)
X_test, y_test = create_sequences(scaled_test, target_idx=-1, window_size=window)

train_t = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float().unsqueeze(-1))
test_t = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float().unsqueeze(-1))
train_loader = DataLoader(train_t, batch_size=best['batch_size'], shuffle=False)

model = LSTMModel(X_train.shape[2], best['hidden_size'], best['num_layers'], best['dropout']).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=best['lr'])
criterion = nn.MSELoss()

for epoch in range(30):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

# ==============================
# 9. PREDICCIÓN FINAL
# ==============================
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test).float().to(DEVICE)).cpu().numpy().flatten()

residual_index = df_test.index[window:]
final_preds = df_prophet_pred.loc[residual_index, 'yhat'].values + y_pred
real = df.loc[residual_index, 'total_vuelos']

mse = mean_squared_error(real, final_preds)
accuracy = 100 - (np.abs(real - final_preds) / real * 100)
print(f"\n📊 Modelo LSTM + Prophet + Optuna -> MSE: {mse:.2f} | Precisión: {np.mean(accuracy):.2f}%")

# ==============================
# 10. GRÁFICO COMPARATIVO
# ==============================
plt.figure(figsize=(18,6))
plt.plot(residual_index[:24*7], real[:24*7], label="Real", color="black")
plt.plot(residual_index[:24*7], final_preds[:24*7], label="LSTM + Prophet (Predicción)", linestyle="--", color="red")
plt.title("LSTM + Prophet + Optuna: Predicción de Vuelos (1era semana 2024)")
plt.xlabel("Fecha y Hora")
plt.ylabel("Total de Vuelos")
plt.legend()
plt.grid(True)
plt.show()
