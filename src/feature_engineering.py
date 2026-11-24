"""Feature engineering time series (lag & rolling) per pasar x komoditas.

Satu fungsi inti (`build_features`) dipakai untuk TRAINING maupun PREDIKSI di website,
sehingga fitur selalu konsisten. Semua fitur historis hanya memakai bulan < bulan target
(shift >= 1), sehingga tidak ada kebocoran data masa depan.
"""
import numpy as np
import pandas as pd
from .config import MIN_OBS_LAST6

# geo_id dibuang: urutan angkanya tidak bermakna. Provinsi (adm1_name) di-one-hot.
CATEGORICAL = ["commodity", "adm1_name"]
NUMERIC = ["year", "month_sin", "month_cos", "ramadan_share", "lebaran_bulan", "pra_lebaran",
           "price_lag_1", "price_lag_2", "price_lag_3", "price_lag_12",
           "rolling_mean_3", "rolling_mean_6", "n_obs_last6", "mom_1", "dev_roll3", "dev_roll6"]
FEATURES = CATEGORICAL + NUMERIC

# Tanggal Ramadan (1 Ramadan) dan Idul Fitri (1 Syawal) per tahun, untuk fitur musiman.
# Tanggal mengikuti penetapan pemerintah Indonesia (Kemenag). 2026 dicek ke pemberitaan sidang isbat:
# 1 Ramadan 19 Februari 2026, 1 Syawal 21 Maret 2026 (versi Muhammadiyah berbeda: 18 Februari dan 20 Maret).
# Tahun 2016–2025 disusun dari penetapan resmi yang sudah diketahui, belum diverifikasi ulang satu per satu.
# Fitur hanya berisi informasi kalender (bukan harga), sehingga aman dari kebocoran data.
RAMADAN_START = {2016: "2016-06-06", 2017: "2017-05-27", 2018: "2018-05-16", 2019: "2019-05-06",
                 2020: "2020-04-24", 2021: "2021-04-13", 2022: "2022-04-02", 2023: "2023-03-23",
                 2024: "2024-03-11", 2025: "2025-03-01", 2026: "2026-02-19"}
EID_FITRI = {2016: "2016-07-06", 2017: "2017-06-25", 2018: "2018-06-15", 2019: "2019-06-05",
             2020: "2020-05-24", 2021: "2021-05-13", 2022: "2022-05-02", 2023: "2023-04-22",
             2024: "2024-04-10", 2025: "2025-03-31", 2026: "2026-03-21"}


def seasonal_features(dates: pd.Series) -> pd.DataFrame:
    """Fitur musiman per bulan kalender (dihitung dari tanggal, bukan dari harga).

    - month_sin / month_cos: posisi bulan dalam setahun (Desember dan Januari berdekatan).
    - ramadan_share: proporsi hari Ramadan dalam bulan tersebut (0..1).
    - lebaran_bulan: 1 jika bulan tersebut memuat Idul Fitri.
    - pra_lebaran: 1 jika bulan sebelum bulan Idul Fitri (permintaan biasanya naik menjelang Lebaran).
    """
    m = dates.dt.month.values
    out = pd.DataFrame(index=dates.index)
    out["month_sin"] = np.sin(2 * np.pi * m / 12)
    out["month_cos"] = np.cos(2 * np.pi * m / 12)
    rs = np.zeros(len(dates)); lb = np.zeros(len(dates)); pl = np.zeros(len(dates))
    yrs = dates.dt.year.values
    for i, (y, mo) in enumerate(zip(yrs, m)):
        month_start = pd.Timestamp(year=int(y), month=int(mo), day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)
        if int(y) in RAMADAN_START:
            r0 = pd.Timestamp(RAMADAN_START[int(y)]); r1 = r0 + pd.Timedelta(days=29)
            overlap = (min(month_end, r1) - max(month_start, r0)).days + 1
            rs[i] = max(overlap, 0) / 30.0
        if int(y) in EID_FITRI:
            e = pd.Timestamp(EID_FITRI[int(y)])
            if month_start <= e <= month_end:
                lb[i] = 1.0
            if month_start + pd.DateOffset(months=1) == e.to_period("M").to_timestamp():
                pl[i] = 1.0
    out["ramadan_share"] = rs
    out["lebaran_bulan"] = lb
    out["pra_lebaran"] = pl
    return out


def build_features(long: pd.DataFrame, extend_months: int = 0) -> pd.DataFrame:
    """Bangun fitur lag/rolling dari `price_clean` (hanya harga observasi valid).

    extend_months: tambah N bulan setelah bulan terakhir (target = NaN) -> dipakai untuk prediksi.
    Kolom hasil: date, geo_id, commodity, adm1_name, target (price_clean), fitur, `eligible`.
    """
    wide = long.pivot_table(index="date", columns=["geo_id", "commodity"], values="price_clean",
                            aggfunc="last", observed=True, dropna=False)
    full_idx = pd.date_range(wide.index.min(), wide.index.max() + pd.DateOffset(months=extend_months),
                             freq="MS")
    wide = wide.reindex(full_idx)
    wide.index.name = "date"

    prev = wide.shift(1)                                   # hanya informasi sebelum bulan target
    feats = {
        "target": wide,
        "price_lag_1": wide.shift(1),
        "price_lag_2": wide.shift(2),
        "price_lag_3": wide.shift(3),
        "price_lag_12": wide.shift(12),
        "rolling_mean_3": prev.rolling(3, min_periods=1).mean(),
        "rolling_mean_6": prev.rolling(6, min_periods=1).mean(),
        "n_obs_last6": prev.notna().astype(float).rolling(6, min_periods=1).sum(),
    }
    frames = {k: v.stack([0, 1], future_stack=True).rename(k) for k, v in feats.items()}
    out = pd.concat(frames.values(), axis=1).reset_index()

    out["mom_1"] = out["price_lag_1"] / out["price_lag_2"] - 1
    out["dev_roll3"] = out["price_lag_1"] / out["rolling_mean_3"] - 1
    out["dev_roll6"] = out["price_lag_1"] / out["rolling_mean_6"] - 1
    out["year"] = out["date"].dt.year
    seas = seasonal_features(out["date"])
    out = pd.concat([out, seas], axis=1)

    meta = long[["geo_id", "adm1_name"]].drop_duplicates("geo_id")
    out = out.merge(meta, on="geo_id", how="left")
    out["geo_id"] = out["geo_id"].astype(str)
    out["adm1_name"] = out["adm1_name"].astype(str)
    out["commodity"] = out["commodity"].astype(str)
    # syarat prediksi yang valid: harga bulan sebelumnya tersedia & riwayat cukup
    out["eligible"] = out["price_lag_1"].notna() & (out["n_obs_last6"] >= MIN_OBS_LAST6)
    return out.sort_values(["commodity", "geo_id", "date"]).reset_index(drop=True)
