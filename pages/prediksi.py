import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.config import COMMODITIES
from src.prediction import (list_models, load_bundle, predict_path, InsufficientHistory, load_metadata,
                            eligible_markets)
from src.ui import (get_long, get_obs, last_obs_month, page_header, idr, fmt_month, fmt_pct, style_fig, show, stat_card,
                    insight, step, read_csv_result, arrow_icon, esc, NAVY)


@st.cache_resource
def _bundle(path):
    return load_bundle(path)


@st.cache_data(show_spinner="Memeriksa kecukupan riwayat tiap pasar...")
def _eligible():
    return eligible_markets(get_long())


def _safe_select(col, label, options, key):
    """selectbox yang aman ketika nilai tersimpan (session_state) tidak ada lagi di opsi baru."""
    if not options:
        st.warning(f"Tidak ada pilihan {label.lower()} untuk filter ini."); st.stop()
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return col.selectbox(label, options, key=key)


def render():
    page_header("Prediksi Harga", "Perkirakan harga bulan-bulan mendatang untuk pasar pilihanmu dalam 3 langkah mudah.")
    obs, long, last = get_obs(), get_long(), last_obs_month()
    meta = load_metadata(); models = list_models(); elig = _eligible()
    nxt = last + pd.DateOffset(months=1)
    st.markdown(f"<div class='pc-note'>Harga asli terakhir yang tercatat: <b>{fmt_month(last)}</b>. "
                f"Prediksi dimulai dari <b>{fmt_month(nxt)}</b>. Hanya pasar dengan riwayat harga yang cukup yang bisa dipilih.</div>",
                unsafe_allow_html=True)

    step(1, "Pilih bahan pokok dan lokasi pasar")
    c = st.columns(4)
    com_name = c[0].selectbox("Bahan pokok", list(COMMODITIES.values()), key="pr_com")
    code = {v: k for k, v in COMMODITIES.items()}[com_name]
    d = obs[(obs.commodity == code) & obs.geo_id.map(lambda g: (g, code) in elig)].drop_duplicates("geo_id")
    if d.empty:
        st.warning("Tidak ada pasar yang memenuhi syarat untuk bahan pokok ini."); return
    # Opsi provinsi/kabupaten/pasar berubah saat bahan pokok diganti. Bila nilai lama tidak ada lagi
    # di opsi baru, Streamlit akan melempar error -> kembalikan ke pilihan pertama secara eksplisit.
    prov_opts = sorted(d.adm1_name.unique())
    prov = _safe_select(c[1], "Provinsi", prov_opts, "pr_prov")
    d = d[d.adm1_name == prov]
    kab_opts = sorted(d.adm2_name.unique())
    kab = _safe_select(c[2], "Kabupaten/Kota", kab_opts, f"pr_kab_{prov}")
    d = d[d.adm2_name == kab]
    pas_opts = sorted(d.mkt_name.unique())
    pas = _safe_select(c[3], "Pasar", pas_opts, f"pr_pas_{prov}_{kab}")
    geo = d[d.mkt_name == pas].geo_id.iloc[0]
    st.caption(f"{len(obs[(obs.commodity == code) & obs.geo_id.map(lambda g: (g, code) in elig)].geo_id.unique())} pasar tersedia untuk {com_name}.")

    step(2, "Lihat hasilnya")
    with st.expander("Pengaturan lanjutan (model)"):
        model_label = st.selectbox("Model yang dipakai", list(models), key="pr_model")
    model_label = st.session_state.get("pr_model", list(models)[0])
    if model_label not in models:            # opsi model bisa berubah setelah retrain
        model_label = list(models)[0]
    # Hanya prediksi 1 bulan ke depan. Horizon 2-3 bulan belum dievaluasi (lihat README).
    if not st.button("Perkirakan harga bulan depan"):
        st.caption("Tekan tombol di atas untuk melihat perkiraan."); return
    bundle = _bundle(models[model_label])
    try:
        path = predict_path(bundle, long, geo, code, 1)
    except InsufficientHistory as e:
        st.markdown(f"<div class='pc-warn'>{e}</div>", unsafe_allow_html=True); return
    r = path[0]; fin = r
    naive = fin["method"] == "naive"
    up = fin["change_vs_observed"] >= 0
    cm = read_csv_result("commodity_method.csv")
    crow = cm[cm["commodity"] == code].iloc[0]
    mape_t = float(crow["mape_test_terpilih"]); cakupan = float(crow["cakupan_uji_pct"])

    if naive:
        sm_text = "Sama dengan harga terakhir: untuk bahan ini model belum lebih akurat dari tebakan harga bulan lalu."
        chg_text = "tetap"
    else:
        chg_text = ("naik" if up else "turun")
        sm_text = (f"{arrow_icon(up)} {chg_text} {idr(abs(fin['change_vs_observed']))} "
                   f"({fmt_pct(abs(fin['change_vs_observed_pct']))}) dibanding harga terakhir "
                   f"{idr(r['last_price'])} ({fmt_month(r['last_date'])})")

    st.markdown(f"<div class='pc-result'><div class='sm'>Perkiraan harga {esc(com_name)} di {esc(pas)} pada {fmt_month(fin['target_date'])}</div>"
                f"<div class='big'>{idr(fin['prediction'])} <span style='font-size:1rem;font-weight:700;color:#E4F7D0'>/kg</span></div>"
                f"<div class='sm'>Kisaran perkiraan (80%): <b>{idr(fin['band_low'])} – {idr(fin['band_high'])}</b> /kg</div>"
                f"<div class='sm'>{sm_text}</div></div>", unsafe_allow_html=True)
    if naive:
        msg = (f"Untuk <b>{esc(com_name)}</b>, model belum terbukti lebih akurat dibanding tebakan harga bulan lalu "
               "pada data validasi, sehingga yang ditampilkan adalah harga terakhir sebagai perkiraan dasar.")
    else:
        msg = (f"Harga <b>{esc(com_name)}</b> di {esc(pas)}, {esc(kab.title())} diperkirakan <b>{chg_text}</b> sekitar "
               f"<b>{idr(abs(fin['change_vs_observed']))}</b> per kg.")
    msg += (f" Pada data uji 2025–2026, perkiraan untuk bahan ini rata-rata meleset sekitar <b>{mape_t:.1f}%</b>. "
            f"Rentang 80% memuat harga sebenarnya pada sekitar <b>{cakupan:.0f}%</b> kasus uji.")
    insight(msg)
    k = st.columns(3)
    with k[0]: stat_card("Harga terakhir tercatat", idr(r["last_price"]), fmt_month(r["last_date"]), "navy")
    with k[1]: stat_card("Perubahan", f"{fin['change_vs_observed']:+,.0f}".replace(",", "."), fmt_pct(fin['change_vs_observed_pct'], True, 2), "navy" if naive else ("up" if up else "down"))
    with k[2]: stat_card("Kecenderungan", "Tetap" if naive else ("Naik" if up else "Turun"), "indikatif, bukan jaminan", "navy" if naive else ("up" if up else "down"))

    h = r["history"].tail(24)
    fig = go.Figure()
    fig.add_scatter(x=h.index, y=h.values, mode="lines+markers", name="Harga tercatat", line=dict(color=NAVY, width=3))
    fig.add_scatter(x=[r["last_date"]] + [p["target_date"] for p in path], y=[r["last_price"]] + [p["prediction"] for p in path],
                    mode="lines+markers", name="Perkiraan", line=dict(color="#E8734A", dash="dash", width=3),
                    marker=dict(size=11, symbol="star"))
    fig.update_layout(title="Harga 24 bulan terakhir dan perkiraan", xaxis_title="Bulan", yaxis_title="Rp/kg")
    show(style_fig(fig, 400))
    with st.expander("Detail teknis model (untuk yang ingin tahu)"):
        st.write(f"**Model:** {bundle['name']} · **Latih ulang:** sampai Desember 2024 · "
                 f"**Uji:** {meta['test_period'][0][:7]} – {meta['test_period'][1][:7]}")
        ft = {k_: v for k_, v in r["features"].items() if k_ not in ("commodity", "geo_id", "adm1_name")}
        st.dataframe(pd.DataFrame({"Fitur": list(ft), "Nilai": list(ft.values())}), hide_index=True, width="stretch")
