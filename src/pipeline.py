"""Pipeline end-to-end: preprocessing -> EDA -> fitur -> training -> evaluasi -> simpan model."""
import json, re
from datetime import datetime
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (MODELS_DIR, RESULTS_DIR, FIG_DIR, FEATURES_CSV, COMMODITIES, SEED)
from .data_loader import load_raw
from .preprocessing import build_long, save_long, last_observed_month
from .eda import make_eda_figures, NAVY, GREEN
from .feature_engineering import build_features, FEATURES
from .evaluation import metrics_table
from . import training as T


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def eval_figures(res, test, preds_test, best_name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cn = COMMODITIES
    t = res[res.split == "test"].copy()
    t["label"] = t["model"].str.replace(" | ", "\n", regex=False)
    t = t.sort_values("MAE")
    for metric, fname in (("MAE", "eval_01_mae.png"), ("RMSE", "eval_02_rmse.png"), ("R2", "eval_03_r2.png")):
        d = t.sort_values(metric, ascending=(metric != "R2"))
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = [GREEN if m == best_name else ("#c0392b" if m == "Naive Forecast" else NAVY) for m in d.model]
        ax.barh(d["model"], d[metric], color=colors)
        ax.invert_yaxis(); ax.set_xlabel(metric + (" (IDR/kg)" if metric != "R2" else ""))
        ax.set_title(f"Perbandingan {metric} pada data test (hijau = model terpilih, merah = baseline naive)")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=130); plt.close(fig)

    y = test["target"].values; p = preds_test[best_name]
    # Actual vs predicted per komoditas
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    rng = np.random.default_rng(SEED)
    for a, (c, n) in zip(ax.ravel(), cn.items()):
        idx = np.where(test["commodity"].values == c)[0]
        idx = rng.choice(idx, min(2500, len(idx)), replace=False)
        a.scatter(y[idx], p[idx], s=4, alpha=.35, color=GREEN)
        lo, hi = min(y[idx].min(), p[idx].min()), max(y[idx].max(), p[idx].max())
        a.plot([lo, hi], [lo, hi], color=NAVY, lw=1)
        a.set_title(n); a.set_xlabel("Aktual (IDR/kg)"); a.set_ylabel("Prediksi (IDR/kg)")
    ax.ravel()[-1].axis("off")
    fig.suptitle(f"Actual vs Predicted (data test) - {best_name}", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eval_04_actual_vs_predicted.png", dpi=130); plt.close(fig)

    # Residual plot
    r = y - p
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].scatter(p, r, s=3, alpha=.25, color=NAVY); ax[0].axhline(0, color="crimson")
    ax[0].set_xlabel("Prediksi (IDR/kg)"); ax[0].set_ylabel("Residual = aktual - prediksi"); ax[0].set_title("Residual vs prediksi")
    ax[1].hist(np.clip(r, -8000, 8000), bins=100, color=GREEN); ax[1].set_xlabel("Residual (IDR/kg, dipotong +/-8000)")
    ax[1].set_ylabel("Frekuensi"); ax[1].set_title("Distribusi residual")
    fig.suptitle(f"Analisis residual - {best_name}", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eval_05_residual.png", dpi=130); plt.close(fig)

    # Global vs separate
    g = t[t.model != "Naive Forecast"].pivot_table(index=["algo", "target_mode"], columns="strategy", values="MAE")
    fig, ax = plt.subplots(figsize=(8, 4.2)); g.plot.bar(ax=ax, color=[GREEN, NAVY], rot=20)
    ax.set_ylabel("MAE test (IDR/kg)"); ax.set_title("Strategi Global vs Separate (MAE data test)"); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eval_06_global_vs_separate.png", dpi=130); plt.close(fig)


def mape(y, p) -> float:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p) / y) * 100)


def commodity_choice(val, test, pv_model, pt_model) -> pd.DataFrame:
    """Per komoditas: pilih 'model' atau 'naive' berdasarkan MAPE VALIDASI (2024).
    Data test hanya dilaporkan, tidak dipakai untuk memilih.

    Juga dihitung:
    - rentang ketidakpastian 80% (kuantil 10% dan 90% dari error relatif validasi),
    - cakupan rentang tersebut pada data test (idealnya mendekati 80%),
    - MAPE metode terpilih per tahun (2024 validasi; 2025 dan 2026 test) sebagai cek stabilitas.
    """
    rows = []
    for c, nama in COMMODITIES.items():
        mv = (val["commodity"] == c).values; mt = (test["commodity"] == c).values
        yv = val["target"].values[mv]; yt = test["target"].values[mt]
        pv_c_model = pv_model[mv]; pv_c_naive = val["price_lag_1"].values[mv]
        pt_c_model = pt_model[mt]; pt_c_naive = test["price_lag_1"].values[mt]
        m_val_model = mape(yv, pv_c_model); m_val_naive = mape(yv, pv_c_naive)
        m_test_model = mape(yt, pt_c_model); m_test_naive = mape(yt, pt_c_naive)
        method = "naive" if m_val_naive <= m_val_model else "model"
        p_val = pv_c_naive if method == "naive" else pv_c_model
        p_test = pt_c_naive if method == "naive" else pt_c_model
        # error relatif validasi -> rentang 80%
        e_val = (yv - p_val) / p_val
        lo_pct, hi_pct = np.quantile(e_val, [0.10, 0.90]) * 100
        lo = p_test * (1 + lo_pct / 100); hi = p_test * (1 + hi_pct / 100)
        coverage = float(np.mean((yt >= lo) & (yt <= hi)) * 100)
        yrs_test = test["date"].dt.year.values[mt]
        yrs_val = val["date"].dt.year.values[mv]
        row = dict(commodity=c, nama=nama, metode=method,
                   mape_val_model=round(m_val_model, 3), mape_val_naive=round(m_val_naive, 3),
                   mape_test_model=round(m_test_model, 3), mape_test_naive=round(m_test_naive, 3),
                   mape_test_terpilih=round(m_test_naive if method == "naive" else m_test_model, 3),
                   band_low_pct=round(lo_pct, 3), band_high_pct=round(hi_pct, 3),
                   cakupan_uji_pct=round(coverage, 2))
        ape_by_year = {}
        for y in np.unique(yrs_val):
            k = yrs_val == y
            ape_by_year[f"mape_{y}"] = round(mape(yv[k], p_val[k]), 3)
        for y in np.unique(yrs_test):
            k = yrs_test == y
            ape_by_year[f"mape_{y}"] = round(mape(yt[k], p_test[k]), 3)
        row.update(ape_by_year)
        rows.append(row)
    return pd.DataFrame(rows)


def run_all(log=print, quick=False):
    for d in (MODELS_DIR, RESULTS_DIR, FIG_DIR, MODELS_DIR / "alternatif"):
        d.mkdir(parents=True, exist_ok=True)

    log("== 1. Muat & preprocessing ==")
    raw = load_raw(); long = build_long(raw); save_long(long)
    log("== 2. EDA ==")
    make_eda_figures(raw, long)
    last_obs = last_observed_month(long)
    log(f"Bulan observasi terakhir: {last_obs.date()}")

    log("== 3. Feature engineering ==")
    feat = build_features(long)
    feat_out = feat[feat["target"].notna() & feat["eligible"]].copy()
    feat_out["date"] = feat_out["date"].dt.strftime("%Y-%m-%d")
    feat_out.to_csv(FEATURES_CSV, index=False, float_format="%.4f")
    log(f"Baris siap modeling: {len(feat_out):,}")

    log("== 4. Training & tuning ==")
    train, val, test, res, pv, pt, store = T.run_experiments(feat, log=log, quick=quick)
    best_name = T.select_best(res)
    log(f"Model terpilih (MAPE validasi terendah): {best_name}")
    log("== 4b. Pilih metode per komoditas (model vs naive, MAPE validasi) ==")
    choice = commodity_choice(val, test, pv[best_name], pt[best_name])
    choice.to_csv(RESULTS_DIR / "commodity_method.csv", index=False)
    commodity_method = dict(zip(choice["commodity"], choice["metode"]))
    commodity_band = {r["commodity"]: (r["band_low_pct"] / 100, r["band_high_pct"] / 100) for _, r in choice.iterrows()}
    for _, r in choice.iterrows():
        log(f"  {r['nama']:<14} metode={r['metode']:<5} MAPE val model={r['mape_val_model']:.2f}% "
            f"naive={r['mape_val_naive']:.2f}% | test terpilih={r['mape_test_terpilih']:.2f}%")

    log("== 5. Evaluasi ==")
    res.to_csv(RESULTS_DIR / "evaluation_metrics.csv", index=False)
    pred_df = test[["date", "geo_id", "commodity", "adm1_name", "target", "price_lag_1"]].rename(
        columns={"target": "y_true"}).reset_index(drop=True)
    for k, v in pt.items():
        pred_df[k] = v
    models_cols = list(pt)
    by_com = metrics_table(pred_df, ["commodity"], pred_cols=models_cols)
    by_com.to_csv(RESULTS_DIR / "evaluation_by_commodity.csv", index=False)
    pred_df["period"] = pred_df["date"].dt.strftime("%Y-%m"); pred_df["year"] = pred_df["date"].dt.year
    metrics_table(pred_df, ["year"], pred_cols=models_cols).to_csv(RESULTS_DIR / "evaluation_by_year.csv", index=False)
    metrics_table(pred_df, ["period"], pred_cols=models_cols).to_csv(RESULTS_DIR / "evaluation_by_period.csv", index=False)
    out = pred_df.drop(columns=["period", "year"]).copy(); out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(RESULTS_DIR / "prediction_results.csv", index=False, float_format="%.2f")
    eval_figures(res, test, pt, best_name)

    log("== 6. Simpan model ==")
    versions = T.library_versions()
    # model terbaik per algoritma (berdasarkan MAPE validasi) -> alternatif untuk website
    v = res[(res.split == "validation") & (res.model != "Naive Forecast")]
    alt = {}
    alt_names = {algo: g.sort_values("MAPE_%").iloc[0]["model"] for algo, g in v.groupby("algo")}
    # Latih ulang model yang dipakai dengan data train + validasi (sampai Desember 2024).
    # Test tetap tidak pernah dipakai untuk melatih; metrik test di metadata berasal dari model train-only.
    log("== 6a. Latih ulang model final dengan data train + validasi (2016-10 s.d. 2024-12) ==")
    full = pd.concat([train, val], ignore_index=True)
    for nm in {best_name, *alt_names.values()}:
        e = store[nm]
        e["models"] = T.fit_strategy(full, e["algo"], e["params"], e["strategy"], e["mode"])
        log(f"  refit: {nm}")
    for algo, nm in alt_names.items():
        bundle = T.make_bundle(store[nm], nm)
        bundle["commodity_method"] = commodity_method
        bundle["commodity_band"] = commodity_band
        fn = MODELS_DIR / "alternatif" / f"{slug(nm)}.pkl"
        joblib.dump(bundle, fn, compress=3)
        alt[T.ALGO_NAMES[store[nm]['algo']] if False else algo] = nm
    best_bundle = T.make_bundle(store[best_name], best_name)
    best_bundle["commodity_method"] = commodity_method
    best_bundle["commodity_band"] = commodity_band
    joblib.dump(best_bundle, MODELS_DIR / "best_model.pkl", compress=3)
    joblib.dump({k: m.named_steps["prep"] for k, m in best_bundle["models"].items()},
                MODELS_DIR / "preprocessing.pkl", compress=3)

    best_rows = res[res.model == best_name].set_index("split")
    naive_rows = res[res.model == "Naive Forecast"].set_index("split")
    meta = {
        "model_name": best_name, "algorithm": best_bundle["algo_name"], "strategy": best_bundle["strategy"],
        "target_mode": best_bundle["target_mode"], "hyperparameters": best_bundle["params"], "seed": SEED,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_period": [str(train.date.min().date()), str(train.date.max().date())],
        "validation_period": [str(val.date.min().date()), str(val.date.max().date())],
        "test_period": [str(test.date.min().date()), str(test.date.max().date())],
        "rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "last_observed_month": str(last_obs.date()),
        "supported_commodities": COMMODITIES, "features": FEATURES,
        "refit_period": ["2016-10-01", str(val.date.max().date())],
        "metrics_test_note": "Metrik test dihitung dari model yang dilatih sampai Desember 2023 (sebelum refit); test tidak dipakai untuk melatih.",
        "selection_rule": ("Model: MAPE validasi (2024) terendah di antara seluruh kombinasi model. "
                           "Per komoditas: model atau naive dipilih berdasarkan MAPE validasi (2024). "
                           "Data test tidak dipakai untuk memilih."),
        "commodity_method": commodity_method,
        "metrics_validation": {k: best_rows.loc["validation", k] for k in ["MAE", "RMSE", "R2", "MAPE_%"]},
        "metrics_test": {k: best_rows.loc["test", k] for k in ["MAE", "RMSE", "R2", "MAPE_%"]},
        "naive_baseline_test": {k: naive_rows.loc["test", k] for k in ["MAE", "RMSE", "R2", "MAPE_%"]},
        "alternative_models": alt, "library_versions": versions,
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float))

    # verifikasi: model yang dimuat ulang menghasilkan prediksi identik
    re_b = joblib.load(MODELS_DIR / "best_model.pkl")
    p2 = T.predict_strategy(re_b["models"], test, re_b["strategy"], re_b["target_mode"])
    p_mem = T.predict_strategy(store[best_name]["models"], test, store[best_name]["strategy"], store[best_name]["mode"])
    assert np.allclose(p2, p_mem), "Prediksi model hasil load tidak konsisten!"
    log("Verifikasi load model: prediksi konsisten (OK)")
    return meta, res
