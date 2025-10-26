# predict_cli.py
import argparse, os, json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from prophet import Prophet
from feature_pipeline import add_time_features, add_residuals, base_feature_list, FESTIVOS_DEFAULT

ARTIF_DIR = "artifacts"

def load_artifacts():
    with open(os.path.join(ARTIF_DIR, "prophet.pkl"), "rb") as f:
        prophet: Prophet = pickle.load(f)
    with open(os.path.join(ARTIF_DIR, "feature_list_base.json"), "r") as f:
        features_base = json.load(f)
    with open(os.path.join(ARTIF_DIR, "feature_list_stacked.json"), "r") as f:
        features_stacked = json.load(f)
    with open(os.path.join(ARTIF_DIR, "festivos.json"), "r") as f:
        festivos = json.load(f)
    with open(os.path.join(ARTIF_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)

    booster6 = xgb.Booster()
    booster6.load_model(os.path.join(ARTIF_DIR, meta["models"]["xgb_6h"]["path"]))
    best_it6 = meta["models"]["xgb_6h"]["best_iteration"]

    booster12 = xgb.Booster()
    booster12.load_model(os.path.join(ARTIF_DIR, meta["models"]["xgb_12h"]["path"]))
    best_it12 = meta["models"]["xgb_12h"]["best_iteration"]

    booster24 = xgb.Booster()
    booster24.load_model(os.path.join(ARTIF_DIR, meta["models"]["xgb_24h"]["path"]))
    best_it24 = meta["models"]["xgb_24h"]["best_iteration"]

    return prophet, features_base, features_stacked, festivos, booster6, best_it6, booster12, best_it12, booster24, best_it24

def fetch_data():
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
    return df

def main():
    parser = argparse.ArgumentParser(description="Inferencia Prophet+XGB (stacked +24h)")
    parser.add_argument("--start", required=True, help="Fecha/hora inicio ISO (ej: 2025-01-01T00:00)")
    parser.add_argument("--hours", type=int, required=True, help="Cantidad de horas a predecir (ej: 24, 168...)")
    args = parser.parse_args()

    start_ts = pd.to_datetime(args.start)
    periods = args.hours

    prophet, features_base, features_stacked, festivos, booster6, best_it6, booster12, best_it12, booster24, best_it24 = load_artifacts()
    df = fetch_data()

    # baseline Prophet para rango requerido (y contexto)
    last_hist = df.index.max()
    end_needed = start_ts + pd.Timedelta(hours=periods-1)

    horizon_extra = int(((end_needed - last_hist).total_seconds()//3600) + 24*7)  # buffer
    future = prophet.make_future_dataframe(periods=max(0, horizon_extra), freq='H')
    forecast = prophet.predict(future).set_index('ds')

    # armamos dataframe de trabajo que incluye histórico + ventana hasta end_needed
    full_idx = pd.date_range(df.index.min(), max(end_needed, df.index.max()), freq='H')
    work = pd.DataFrame(index=full_idx)
    work['total_vuelos'] = df['total_vuelos'].reindex(full_idx)
    work['prophet_pred'] = forecast['yhat'].reindex(full_idx).interpolate()

    # features/ residual/ lags/ rolling
    work = add_time_features(work, festivos)
    work = work.assign(residual = work['total_vuelos'] - work['prophet_pred'])
    for lag in [1,6,12,24]:
        work[f'lag_{lag}h'] = work['residual'].shift(lag)
    for r in [6,12,24]:
        work[f'rolling_{r}h'] = work['residual'].shift(1).rolling(r).mean()

    # pred_6h y pred_12h (predicciones out-of-sample sobre todo el índice disponible)
    dm_base = xgb.DMatrix(work[features_base].astype(np.float32).values)
    work['pred_6h']  = booster6.predict(dm_base, iteration_range=(0, best_it6+1))
    work['pred_12h'] = booster12.predict(dm_base, iteration_range=(0, best_it12+1))

    # predicción stacked +24h
    X_stacked = work[features_stacked].astype(np.float32)
    dm_stacked = xgb.DMatrix(X_stacked.values)
    resid_24h_pred = booster24.predict(dm_stacked, iteration_range=(0, best_it24+1))

    # reconstrucción final: prophet + residuo_pred (+24h aplicado por alineación)
    # Para horizonte: tomar las filas desde start_ts a end_needed
    pred_series = pd.Series(resid_24h_pred, index=work.index) + work['prophet_pred']
    pred_out = pred_series.loc[start_ts:end_needed]

    # salida
    print("\n✅ Predicciones")
    print(pred_out.head(10))
    out_df = pd.DataFrame({"yhat_final": pred_out})
    out_path = os.path.join(ARTIF_DIR, f"pred_{start_ts:%Y%m%dT%H}_{periods}h.csv")
    out_df.to_csv(out_path)
    print(f"\n💾 Guardado: {out_path}")

if __name__ == "__main__":
    main()
