"""Uji kebijakan model PasarCerdas: python tests/test_model_policy.py

Memeriksa aturan yang dibuat pada pembaruan terakhir:
- geo_id tidak lagi dipakai sebagai fitur; fitur musiman tersedia.
- fitur kalender Ramadan/Lebaran dihitung benar.
- metadata dan file hasil konsisten dengan fitur dan komoditas.
- komoditas yang memakai metode naif benar-benar memprediksi harga terakhir.
- rentang ketidakpastian valid dan cakupannya masuk akal.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib

ok = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name, extra); ok.append(bool(cond))

from src.config import COMMODITIES, MODELS_DIR, RESULTS_DIR
from src.feature_engineering import FEATURES, CATEGORICAL, NUMERIC, seasonal_features

# 1. fitur
check("geo_id bukan fitur", "geo_id" not in FEATURES)
check("fitur musiman ada", all(c in NUMERIC for c in ["month_sin", "month_cos", "ramadan_share", "lebaran_bulan", "pra_lebaran"]))
check("month mentah tidak dipakai lagi", "month" not in NUMERIC)

# 2. fitur kalender
d = pd.Series(pd.to_datetime(["2024-03-01", "2024-04-01", "2025-03-01", "2025-02-01"]))
s = seasonal_features(d)
check("Lebaran 2024 jatuh di April", s.loc[1, "lebaran_bulan"] == 1 and s.loc[0, "lebaran_bulan"] == 0)
check("Maret 2024 = pra-Lebaran", s.loc[0, "pra_lebaran"] == 1)
check("Ramadan 2024: Maret memuat sebagian besar hari puasa", 0.5 < s.loc[0, "ramadan_share"] <= 1.0,
      f"(share={s.loc[0, 'ramadan_share']:.2f})")
check("Februari 2025 = pra-Lebaran (Lebaran Maret 2025)", s.loc[3, "pra_lebaran"] == 1)
check("siklus bulan: Des dan Jan berdekatan", np.allclose(
    seasonal_features(pd.Series(pd.to_datetime(["2024-12-01"])))[["month_sin", "month_cos"]].values,
    seasonal_features(pd.Series(pd.to_datetime(["2024-01-01"])))[["month_sin", "month_cos"]].values, atol=0.9))

d26 = pd.Series(pd.to_datetime(["2026-02-01", "2026-03-01"]))
s26 = seasonal_features(d26)
check("Ramadan 2026 mulai 19 Februari (Kemenag): Februari 10/30 hari", abs(s26.loc[0, "ramadan_share"] - 10/30) < 1e-6)
check("Idul Fitri 2026 (21 Maret) jatuh di Maret", s26.loc[1, "lebaran_bulan"] == 1)
check("Februari 2026 = pra-Lebaran", s26.loc[0, "pra_lebaran"] == 1)

# 3. metadata & hasil
meta = json.load(open(MODELS_DIR / "model_metadata.json", encoding="utf-8"))
check("metadata.features sama dengan FEATURES", meta["features"] == FEATURES)
check("metadata mencatat refit sampai 2024", str(meta.get("refit_period", [""])[1]).startswith("2024-12"))
check("metadata mencatat metrik test berasal dari model sebelum refit", "metrics_test_note" in meta)
cm = pd.read_csv(RESULTS_DIR / "commodity_method.csv")
need = {"commodity", "metode", "mape_test_terpilih", "band_low_pct", "band_high_pct", "cakupan_uji_pct"}
check("commodity_method.csv punya kolom yang dibutuhkan", need.issubset(cm.columns))
check("semua komoditas ada", set(cm["commodity"]) == set(COMMODITIES))
check("metode hanya 'model' atau 'naive'", set(cm["metode"]).issubset({"model", "naive"}))
check("rentang valid (batas bawah <= batas atas)", (cm["band_low_pct"] <= cm["band_high_pct"]).all())
check("cakupan rentang di data uji 50–100%", cm["cakupan_uji_pct"].between(50, 100).all(),
      f"(min={cm['cakupan_uji_pct'].min():.1f}, max={cm['cakupan_uji_pct'].max():.1f})")

# 4. bundle konsisten dengan hasil
b = joblib.load(MODELS_DIR / "best_model.pkl")
check("bundle punya metode per komoditas", set(b["commodity_method"]) == set(COMMODITIES))
check("bundle punya rentang per komoditas", set(b["commodity_band"]) == set(COMMODITIES))
check("bundle.features sama dengan FEATURES", b["features"] == FEATURES)
check("metode di bundle sama dengan CSV", all(b["commodity_method"][r.commodity] == r.metode for r in cm.itertuples()))

# 5. prediksi naif = harga terakhir; model = tidak negatif
from src.prediction import load_bundle, predict_path
from src.data_loader import load_long
long = load_long()
last = long[(long.date == "2026-09-01") & long.price_clean.notna()]
for c in COMMODITIES:
    cands = last[last.commodity == c]
    if cands.empty:
        continue
    r = cands.iloc[0]
    try:
        out = predict_path(b, long, r.geo_id, c, 1)[0]
    except Exception as e:
        check(f"prediksi {c} berjalan", False, str(e)); continue
    if b["commodity_method"][c] == "naive":
        check(f"naif {c}: prediksi = harga terakhir", abs(out["prediction"] - out["last_price"]) < 1e-6)
    else:
        check(f"model {c}: prediksi positif", out["prediction"] > 0)
    check(f"rentang {c}: batas bawah <= batas atas", out["band_low"] <= out["band_high"])

print(f"\nHASIL: {sum(ok)}/{len(ok)} lulus")
sys.exit(0 if all(ok) else 1)
