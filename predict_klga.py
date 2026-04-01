#!/usr/bin/env python3
import argparse, json, base64, pickle, warnings, os, sys, traceback
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

PORT = os.environ.get("PORT", "5177")
BASE = f"http://127.0.0.1:{PORT}"
LAT, LON = 40.7769, -73.8740

def solar_factor(ts_utc_series):
    t = pd.to_datetime(ts_utc_series, utc=True)
    doy = t.dt.dayofyear.values.astype(float)
    frac_hour = (t.dt.hour.values + t.dt.minute.values/60.0)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (frac_hour - 12.0)/24.0)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma))
    eqt = (229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)))
    ha = np.deg2rad((((frac_hour*60.0 + eqt + 4.0 * LON)/4.0) - 180.0))
    elev = np.arcsin(np.sin(np.deg2rad(LAT))*np.sin(decl) + np.cos(np.deg2rad(LAT))*np.cos(decl)*np.cos(ha))
    return pd.Series(np.clip(np.sin(elev), 0, 1), index=ts_utc_series.index)

def get_arr(obs, key):
    for k in obs.keys():
        if k.startswith(key): return obs[k]
    return [None] * len(obs.get("date_time", []))

def process_station_df(obs, prefix=""):
    df = pd.DataFrame({
        "ts": pd.to_datetime(obs["date_time"], utc=True), 
        f"{prefix}temp_f": pd.to_numeric(get_arr(obs, "air_temp"), errors="coerce"),
        f"{prefix}dew_f": pd.to_numeric(get_arr(obs, "dew_point_temperature"), errors="coerce"),
        f"{prefix}wind_mph": pd.to_numeric(get_arr(obs, "wind_speed"), errors="coerce"),
        f"{prefix}wind_dir": pd.to_numeric(get_arr(obs, "wind_direction"), errors="coerce"),
        f"{prefix}alt_inhg": pd.to_numeric(get_arr(obs, "altimeter"), errors="coerce"),
    })
    return df.set_index("ts").resample("1min").ffill(limit=15).reset_index().dropna(subset=[f"{prefix}temp_f"])

def main():
    with open("model_klga_tri.json", "r") as f: m = json.load(f)
    
    r_target = requests.get(f"{BASE}/api/hf_96", params={"station": m["station"]}, timeout=15)
    r_lead = requests.get(f"{BASE}/api/hf_96", params={"station": m["lead_station"]}, timeout=15)
    r_north = requests.get(f"{BASE}/api/hf_96", params={"station": m["north_station"]}, timeout=15)
    
    df = process_station_df(r_target.json()["STATION"][0]["OBSERVATIONS"])
    df = pd.merge(df, process_station_df(r_lead.json()["STATION"][0]["OBSERVATIONS"], prefix="kewr_"), on="ts", how="left")
    df = pd.merge(df, process_station_df(r_north.json()["STATION"][0]["OBSERVATIONS"], prefix="kteb_"), on="ts", how="left")
    df = df.ffill().bfill()
    
    pt = df["ts"].dt.tz_convert("America/New_York")
    t_c, d_c = (df["temp_f"]-32)*5/9, (df["dew_f"]-32)*5/9
    df["rh"] = 100 * np.exp((17.625*d_c)/(243.04+d_c)) / np.exp((17.625*t_c)/(243.04+t_c))
    df["sf"] = solar_factor(df["ts"])
    df["sf_lag_30"] = df["sf"].shift(30).fillna(0)
    df["alt_d_30"] = df["alt_inhg"] - df["alt_inhg"].shift(30)
    df["temp_v_15"] = df["temp_f"].rolling(15).std().fillna(0)
    
    df["temp_lag_30_diff"] = df["temp_f"] - df["temp_f"].shift(30)
    df["temp_lag_60_diff"] = df["temp_f"] - df["temp_f"].shift(60)
    
    for pfx in ["", "kewr_", "kteb_"]:
        spd = df[f"{pfx}wind_mph"].fillna(0)
        dir_rad = np.radians(df[f"{pfx}wind_dir"].fillna(0))
        df[f"{pfx}wind_u"] = spd * np.sin(dir_rad)
        df[f"{pfx}wind_v"] = spd * np.cos(dir_rad)
        
        if pfx != "":
            df[f"{pfx}temp_diff"] = df[f"{pfx}temp_f"] - df["temp_f"]
            df[f"{pfx}wind_surge"] = df[f"{pfx}wind_mph"] - df[f"{pfx}wind_mph"].shift(30)
    
    frac = pt.dt.hour + pt.dt.minute/60.0
    df["h_sin"], df["h_cos"] = np.sin(2*np.pi*frac/24.0), np.cos(2*np.pi*frac/24.0)
    
    df2 = df.dropna(subset=m["feature_cols"])
    row = df2.iloc[[-1]]
    out = {"ts": row["ts"].iloc[0].isoformat(), "temp_now_f": float(row["temp_f"].iloc[0]), "forecasts": {}}
    
    baseline_maes = {30: 1.13, 60: 1.43, 120: 1.81}
    for h in m["horizons"]:
        reg = pickle.loads(base64.b64decode(m["regs"][str(h)]))
        delta = float(reg.predict(row[m["feature_cols"]])[0])
        
        base_err = baseline_maes.get(h, 1.5)
        volatility_penalty = float(row["temp_v_15"].iloc[0]) * 0.75 
        dynamic_err = base_err + volatility_penalty
        confidence = max(0.0, min(100.0, 100.0 - (dynamic_err * 12)))
        
        if delta > 0.5: trend = "up"
        elif delta < -0.5: trend = "down"
        else: trend = "flat"
            
        out["forecasts"][str(h)] = {
            "temp_pred_f": out["temp_now_f"] + delta, 
            "delta_pred_f": delta, 
            "exp_abs_err_f": round(dynamic_err, 2), 
            "trend": trend, 
            "confidence": round(confidence, 1)
        }

    hist_df = df2.tail(480).copy()
    if not hist_df.empty and "120" in m["regs"]:
        reg120 = pickle.loads(base64.b64decode(m["regs"]["120"]))
        hist_deltas = reg120.predict(hist_df[m["feature_cols"]])
        out["historical_120"] = [{"ts": (t + pd.Timedelta(minutes=120)).isoformat(), "temp_pred_f": tmp + d} for t, tmp, d in zip(hist_df["ts"], hist_df["temp_f"], hist_deltas)]

    reg_high = pickle.loads(base64.b64decode(m["reg_high"]))
    delta_high = float(reg_high.predict(row[m["feature_cols"]])[0])
    
    df_today = df[pt.dt.date == pt.iloc[-1].date()]
    df_51 = df_today[df_today["ts"].dt.minute == 51]
    high_so_far = float(df_51["temp_f"].max()) if not df_51.empty else float(df_today["temp_f"].max())
    
    out["expected_high"] = {"temp_pred_f": max(high_so_far, out["temp_now_f"] + delta_high), "temp_high_so_far": high_so_far}
    out["drivers"] = ["Tri-Station ML (KLGA + KEWR + KTEB)"]
    print(json.dumps(out))

if __name__ == "__main__":
    try: main()
    except Exception as e: print(json.dumps({"error": f"Python Error: {str(e)}", "trace": traceback.format_exc()})); sys.exit(0)