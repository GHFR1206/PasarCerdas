# Catatan Perbaikan — PasarCerdas

Dokumen ini mencatat seluruh analisis dan perbaikan yang dilakukan pada repository
[ctrl-stiwww/PasarCerdas](https://github.com/ctrl-stiwww/PasarCerdas).

- **Tanggal:** 25 November 2026
- **Basis commit:** `1a2ea52` (perbaiki results dan models)
- **Commit hasil perbaikan:** `8035dcc`, `b00d53b`, `1bc4970` + commit dokumen ini
- **Status:** sudah di-commit lokal, **belum di-push** ke remote
- **Lingkungan verifikasi:** Python 3.14.10 (venv), pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, xgboost 3.2.0, streamlit 1.67.0

---

## 1. Ringkasan

Repo sudah berfungsi dan seluruh test bawaan lulus, tetapi ada beberapa **bug fungsional yang
tidak tertangkap test** — terutama filter tanggal yang diam-diam diabaikan, format Rupiah yang
salah, dan satu sel verifikasi di notebook yang selalu mencetak `False`.

Total: **15 butir perbaikan** pada 20 file (kode, test, dokumentasi, konfigurasi),
plus 2 file turunan yang dikeluarkan dari Git.

| Kategori | Butir |
|---|---|
| Bug fungsional (perilaku salah) | 4 — format Rupiah, filter tanggal, selectbox bertingkat, nama wilayah |
| Bug verifikasi / konten menyesatkan | 3 — notebook 05, insight dashboard, klaim evaluasi |
| Performa | 1 — `eligible_markets` |
| Ketahanan & pesan error | 2 — escape HTML, pesan `predict_path` |
| Kebersihan repo & dependensi | 2 — file turunan, requirements |
| Kode mati & dokumentasi | 2 — pembersihan `pyflakes`, README |
| Revamp tampilan (poles frontend) | 1 — design system, panel kartu, menu aktif, komponen baru |

**Yang tidak berubah:** model terpilih tetap `Random Forest | global | ratio`, MAPE uji tetap
**5,18%**, dan 6 dari 7 komoditas tetap memakai naive. Perbaikan ini soal kebenaran kode dan UI,
bukan peningkatan akurasi.

---

## 2. Metode analisis

1. Clone repo, baca seluruh modul (`src/`, `pages/`, `tests/`, `retrain.py`) dan notebook.
2. Pasang dependensi di venv terisolasi, jalankan test bawaan → **18/18 dan 37/37 lulus**.
3. Probe manual untuk mencari perilaku yang tidak tercakup test:
   - Streamlit `AppTest` untuk simulasi klik/filter di semua halaman.
   - Cek invarian data (kecocokan `eligible_markets` lama vs baru, konsistensi metrik).
   - `pyflakes` untuk import/variabel mati.
4. Verifikasi tiap dugaan **sebelum** memperbaiki (tidak ada perbaikan spekulatif).
5. Setelah perbaikan: jalankan ulang test, eksekusi kelima notebook, jalankan `retrain.py`
   penuh di salinan terpisah, dan jalankan aplikasi dari clone bersih.

---

## 3. Daftar perbaikan

### 3.1 Format Rupiah salah untuk angka 1 desimal

| | |
|---|---|
| **File** | `src/ui.py` |
| **Tingkat** | Bug fungsional (tampilan angka) |

**Masalah.** Fungsi `idr()` memakai `f"{x:,.{d}f}".replace(",", ".")`. Untuk `d=1`, hasilnya
`"1,234.6"` → setelah replace menjadi `"1.234.6"` — titik dipakai sebagai pemisah ribuan **dan**
desimal sekaligus, sehingga angkanya salah baca.

**Sebelum:**
```python
def idr(x, d=0):
    return "-" if x is None or pd.isna(x) else "Rp " + f"{x:,.{d}f}".replace(",", ".")
```
```text
idr(1234.56, 1)  ->  "Rp 1.234.6"   # salah: seharusnya Rp 1.234,6
```

**Sesudah.** Pemisah ditukar dengan placeholder agar tidak saling menimpa, plus guard NaN.
```python
def idr(x, d=0):
    if x is None or pd.isna(x):
        return "-"
    s = f"{x:,.{d}f}"
    s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return "Rp " + s
```
```text
idr(1234567)      -> "Rp 1.234.567"
idr(1234.56, 1)   -> "Rp 1.234,6"
idr(1234.567, 2)  -> "Rp 1.234,57"
```

`fmt_pct()` juga ditambah guard NaN agar tidak mencetak `"nan%"`.

---

### 3.2 Filter "Periode" di Cek Harga diabaikan saat memilih satu tanggal

| | |
|---|---|
| **File** | `pages/harga_pangan.py` |
| **Tingkat** | Bug fungsional — **paling serius** |

**Masalah.** `st.date_input()` mengembalikan `tuple` bila rentang, tetapi mengembalikan **satu
`date`** bila pengguna memilih satu hari. Kode hanya menangani `tuple`:

```python
if isinstance(rng, tuple) and len(rng) == 2:
    d = d[(d.date >= pd.Timestamp(rng[0])) & (d.date <= pd.Timestamp(rng[1]))]
```

Akibatnya, memilih satu tanggal **tidak menyaring apa pun** — semua data tetap tampil, tanpa
pesan error. Pengguna mengira sudah memfilter padahal belum.

**Sesudah.** Menangani 2 tanggal, 1 tanggal, dan input kosong.
```python
if isinstance(rng, (tuple, list)) and len(rng) == 2:
    start, end = rng
elif isinstance(rng, (tuple, list)) and len(rng) == 1:
    start = end = rng[0]
elif rng is not None:
    start = end = rng
else:
    start, end = dmin, dmax
d = d[(d.date >= pd.Timestamp(start)) & (d.date <= pd.Timestamp(end))]
```

Ditambahkan test regresi: memilih 1 tanggal harus menghasilkan tepat 1 baris data per bulan.

---

### 3.3 Halaman error saat mengganti komoditas setelah memilih wilayah

| | |
|---|---|
| **File** | `pages/harga_pangan.py`, `pages/prediksi.py`, `pages/analisis_tren.py` |
| **Tingkat** | Bug fungsional (crash) |

**Masalah.** Filter bertingkat (Komoditas → Provinsi → Kabupaten → Pasar) menyimpan pilihan di
`st.session_state`. Ketika komoditas diganti, daftar opsi menyusut dan nilai lama bisa tidak ada
lagi. Streamlit melempar `StreamlitAPIException` karena nilai widget tidak ada di `options`.

**Sesudah.** Helper yang memvalidasi nilai tersimpan sebelum widget dirender:
```python
def _safe_select(col, label, options, key):
    if not options:
        st.warning(f"Tidak ada pilihan {label.lower()} untuk filter ini."); st.stop()
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return col.selectbox(label, options, key=key)
```
Dipakai di ketiga halaman. Di halaman Prediksi juga ditambah validasi label model dari
`session_state` (bisa hilang setelah retrain) dengan fallback ke model pertama.

Ditambahkan test regresi untuk skenario: pilih `Telur → NUSA TENGGARA BARAT → SUMBAWA`,
lalu ganti ke `Beras`.

---

### 3.4 `predict_path` memberi pesan keliru untuk pasar tanpa observasi di bulan terakhir

| | |
|---|---|
| **File** | `src/prediction.py` |
| **Tingkat** | Perbaikan pesan error (bukan crash) |

**Klarifikasi penting.** Saat analisis awal aku menduga ini crash, tetapi setelah diuji ulang
secara menyeluruh (**400 kombinasi pasar–komoditas, horizon 3 langkah**) versi asli **tidak
pernah crash**: 381 sukses, 20 melempar `InsufficientHistory` yang tertangani, **0 crash**.
Dugaan awal itu keliru dan sudah dikoreksi di sini.

**Masalah sebenarnya.** Untuk pasar yang observasinya berhenti sebelum bulan terakhir data,
`_predict_from()` menghitung `last_obs = sub["date"].max()` (bulan terakhir **pasar itu**,
mis. Agustus 2026), lalu pesannya menyebut *"tidak memiliki harga observasi pada Agustus 2026
(bulan terakhir data)"* — padahal bulan terakhir **dataset** adalah September 2026. Pesannya
membingungkan: pengguna melihat harga Agustus ada di tabel, tapi diberi tahu "tidak ada".

**Sesudah.** Validasi eksplisit di awal `predict_path()` memakai bulan terakhir dataset, dengan
pesan yang menyebut bulan yang benar:
```python
if sub.empty or pd.isna(sub.loc[sub["date"] == last_obs, "price_clean"]).all():
    raise InsufficientHistory(
        f"Pasar ini tidak memiliki harga observasi pada bulan terakhir data "
        f"({last_obs:%B %Y}), sehingga prediksi bulan berikutnya tidak dapat dibuat tanpa "
        "mengisi data secara sembarangan.")
```
Perilaku (raise `InsufficientHistory`) tetap sama seperti sebelumnya — yang berubah hanya
kejelasan dan ketepatan pesannya, plus pengecekan dilakukan sekali di depan alih-alih di dalam loop.

---

### 3.5 `eligible_markets` lambat (25,3 detik → 0,012 detik)

| | |
|---|---|
| **File** | `src/prediction.py` |
| **Tingkat** | Performa |

**Masalah.** Untuk sekadar mendapat daftar pasar yang layak diprediksi, fungsi ini membangun
**seluruh tabel fitur** (371 ribu baris, `pivot_table` + `stack`) lalu memfilternya.
Waktu terukur: **25,3 detik**, dijalankan setiap kali halaman Prediksi dibuka (di-cache, tapi
tetap mahal saat cold start).

**Sesudah.** Syarat yang sama dihitung langsung dari data observasi:
```python
last_obs = last_observed_month(long)
window = long[(long["date"] > last_obs - pd.DateOffset(months=6)) & (long["date"] <= last_obs)]
counts = window.groupby(["geo_id", "commodity"], observed=True)["price_clean"].count()
n_obs = {(str(g), str(c)): int(v) for (g, c), v in counts.items()}
has_last = window[(window["date"] == last_obs) & window["price_clean"].notna()]
keys = set(zip(has_last["geo_id"].astype(str), has_last["commodity"].astype(str)))
return {k for k in keys if n_obs.get(k, 0) >= MIN_OBS_LAST6}
```

**Bukti kesetaraan** (diuji langsung, bukan diasumsikan):
```text
versi lama: 25,3 detik, 1483 pasangan
versi baru:  0,012 detik, 1483 pasangan
identik  : True   (selisih himpunan = kosong di kedua arah)
```
Perbaikan ~2.100× lebih cepat, dengan hasil yang terbukti identik.

---

### 3.6 Notebook 05 selalu mencetak `Konsisten: False`

| | |
|---|---|
| **File** | `notebooks/05_model_evaluation.ipynb` |
| **Tingkat** | Bug verifikasi (menyesatkan) |

**Masalah.** Output yang tersimpan di notebook berisi:
```text
Konsisten: False | NaN: 0
```
Penyebabnya: sel membandingkan prediksi dari **bundle hasil refit** (dilatih sampai Des 2024)
dengan `results/prediction_results.csv`, yang berisi prediksi dari model **sebelum refit**
(hanya sampai Des 2023). Perbandingan apel-vs-jeruk, jadi wajar selalu `False`.

**Sesudah.** Sel diverifikasi ulang dan dipecah menjadi dua pemeriksaan yang benar:
1. Bundle final memang hasil refit — prediksinya pada data uji *wajar* berbeda dari metrik metadata.
2. Metrik test di `model_metadata.json` dapat direproduksi dari model train-only dengan
   hyperparameter yang sama.

Hasil eksekusi baru:
```text
Bundle refit di data test: {'n': 27798, 'MAE': 1893.487, 'RMSE': 3657.252, 'R2': 0.906, 'MAPE_%': 5.111}
Reproduksi metrik test metadata -> selisih MAPE: 0.001 %
Prediksi valid (finite & > 0): True
```

Artinya metrik di metadata memang **out-of-sample** dan dapat dipertanggungjawabkan.

---

### 3.7 Insight dashboard menyesatkan saat semua harga naik

| | |
|---|---|
| **File** | `pages/dashboard.py`, `src/ui.py` |
| **Tingkat** | Bug tampilan / teks salah |

**Masalah.** Kode selalu mengambil elemen terakhir dan pertama dari daftar terurut, lalu menulis
"paling naik" dan "paling turun" — tanpa memeriksa tanda. Bila semua harga naik, komoditas
dengan kenaikan terkecil tetap dilabeli "paling turun". Kartu sorotan (`movers_card`) punya
masalah sama.

**Sesudah.** Tiga cabang eksplisit:
- ada yang naik dan ada yang turun → "paling naik" / "paling turun";
- semua naik → "kenaikan terbesar" / "kenaikan terkecil";
- semua turun → "penurunan terbesar" / "penurunan terkecil".

Ditambah guard: bila sebuah komoditas tidak punya pasangan harga di kedua bulan, komoditas itu
dilewati (halaman tidak crash) dan bila tidak ada data sama sekali muncul peringatan jelas.

---

### 3.8 Nama wilayah resmi dirusak `.title()`

| | |
|---|---|
| **File** | `pages/perbandingan.py`, `pages/prediksi.py` |
| **Tingkat** | Bug tampilan (data salah tampil) |

**Masalah.** `.title()` diterapkan pada nama wilayah dari dataset:
```text
KOTA BAU-BAU  ->  Kota Bau-Bau
DKI JAKARTA   ->  Dki Jakarta
```
Nama resmi jadi salah ejaan.

**Sesudah.** Nama ditampilkan apa adanya (dataset sudah memakai huruf kapital konsisten).
Di `perbandingan.py` sekaligus diperbaiki judul tabel yang hardcode `"10 pasar termurah"`
padahal slider mengizinkan 5–20 — sekarang mengikuti nilai slider.

---

### 3.9 Klaim evaluasi model ditulis mati (hardcoded)

| | |
|---|---|
| **File** | `pages/evaluasi_model.py` |
| **Tingkat** | Bug konten (tidak mengikuti data) |

**Masalah.** Kalimat *"Keunggulan model terlihat pada RMSE/R² dan komoditas bawang merah"*
ditulis statis. Kalau data berubah (retrain dengan dataset baru), kalimat ini bisa jadi bohong.

**Sesudah.** Klaim dihitung dari `commodity_method.csv` — menyebut komoditas mana yang benar-benar
memakai metode model, dan menyatakan secara eksplisit bila model belum mengungguli baseline.

---

### 3.10 Escape HTML untuk teks dari dataset

| | |
|---|---|
| **File** | `src/ui.py`, `pages/dashboard.py`, `pages/prediksi.py` |
| **Tingkat** | Ketahanan (robustness) |

Nama pasar/provinsi/kabupaten berasal dari dataset dan disisipkan ke `st.markdown(...,
unsafe_allow_html=True)`. Bila suatu saat dataset memuat karakter `<`, `>`, `&`, tata letak bisa
rusak atau HTML ikut tereksekusi. Ditambahkan `esc()` dan diterapkan pada semua teks dinamis
di komponen kartu dan blok hasil prediksi.

---

### 3.11 Repo membengkak oleh file turunan

| | |
|---|---|
| **File** | `.gitignore`, Git index |
| **Tingkat** | Kebersihan repo |

**Masalah.**
- `dataset/processed/features_model.csv` (17,7 MB) sudah ada di `.gitignore` **tetapi masih
  ter-track** — `.gitignore` tidak berpengaruh pada file yang sudah di-commit.
- `results/prediction_results.csv` (5,1 MB, dump seluruh data uji) ikut ter-commit padahal selalu
  dibuat ulang oleh `retrain.py`.

**Sesudah.** Keduanya dikeluarkan dari Git (`git rm --cached`) dan masuk `.gitignore`.
Notebook 05 disesuaikan sehingga **tidak lagi membaca** `prediction_results.csv`, jadi tetap
bisa dijalankan setelah clone baru (sudah diuji).

Dampak: **−132.260 baris** dari repo (104.461 + 27.799 baris). `prices_long.csv` dan file model
`.pkl` **tetap** di-commit karena dibutuhkan saat aplikasi jalan di Streamlit Community Cloud.

---

### 3.12 Dependensi rapuh

| | |
|---|---|
| **File** | `requirements.txt`, `requirements-dev.txt` (baru) |
| **Tingkat** | Reproduksibilitas |

**Masalah.** `scikit-learn==1.9.1` dan `xgboost==3.2.0` dipin (benar, karena `.pkl` butuh versi
itu), tetapi `numpy` dan `pandas` dibiarkan tanpa batas atas — padahal kode memakai API yang
bisa berubah di versi mayor baru. Dependensi notebook juga dicampur ke requirements aplikasi,
sehingga `pip install -r requirements.txt` di server menarik Jupyter, nbconvert, dll.

**Sesudah.**
```text
numpy>=2.0,<3
pandas>=2.2,<4
scikit-learn==1.9.1      # dipin persis: .pkl dibuat dengan versi ini
xgboost==3.2.0           # dipin persis
joblib>=1.4,<2
matplotlib>=3.8,<4
plotly>=5.24,<8
streamlit>=1.50,<2
```
Dependensi notebook dipindah ke `requirements-dev.txt`.

---

### 3.13 Pembersihan kode mati

`pyflakes` menemukan **24 temuan** import/variabel tak terpakai. Yang dihapus antara lain:

| File | Yang dibuang |
|---|---|
| `src/training.py` | `OrdinalEncoder`, `datetime`, `metrics_table`, `MODELS_DIR`, `RESULTS_DIR` |
| `src/prediction.py` | `Path`, `COMMODITIES`, `FEATURES` |
| `src/preprocessing.py` | konstanta `ID_COLS` (tidak pernah dipakai) |
| `src/pipeline.py` | `TRAIN_END`, `VAL_START`, `VAL_END`, `TEST_START`, variabel `first` |
| `src/data_loader.py` | `RAW_CSV_NAME` |
| `src/ui.py` | `COMMODITIES` |
| `pages/*` | `json`, `fmt_month`, `help_box`, `GREEN`, variabel `last` |
| `retrain.py` | `pathlib.Path` |

Status akhir: `pyflakes` **bersih (exit 0)**.

---

### 3.15 Revamp tampilan (poles frontend)

Pekerjaan lanjutan atas permintaan "poles dan perbagus" — **tidak mengubah logika data,
model, maupun angka evaluasi**. Semua perubahan ada di lapisan presentasi.

**a. Design system sebagai token.** `assets/style.css` ditulis ulang dengan variabel CSS:
warna merek, warna semantik naik/turun (`--up`/`--down` beserta latar & garis), skala tipografi,
radius (`--r-sm` s/d `--r-xl`), bayangan berlapis (`--shadow-sm/--shadow/--shadow-lift`),
dan `--ease`. Tujuannya agar konsisten dan mudah diubah dari satu tempat.

**b. Token tema native Streamlit.** `.streamlit/config.toml` kini memakai opsi tema resmi
(`font`, `headingFont`, `baseFontSize`, `headingFontSizes`, `headingFontWeights`,
`metricValueFontSize`, `linkColor`, `borderColor`, `chartCategoricalColors`) sehingga widget
dan grafik ikut konsisten, bukan hanya elemen yang di-`markdown` manual.

**c. Bug styling yang ditemukan saat verifikasi visual.** Streamlit 1.67 **sudah tidak memakai
BaseWeb** untuk selectbox/tab/pills. Selector lama (`div[data-baseweb="select"] > div`,
`button[data-baseweb="tab"]`, `div[data-testid="stPills"]`) **tidak pernah cocok** sehingga
kustomisasi input dan tab selama ini tidak berefek. Diganti selector `data-testid` yang benar:
`[data-testid="stSelectbox"] [role="group"]`, `[data-testid="stDateInputField"]`,
`[data-testid="stTab"]`, `[data-testid="stButtonGroup"] button[aria-pressed="true"]`,
`[role="listbox"]`/`[role="option"]`.

**d. Menu aktif tidak pernah tersorot.** Streamlit tidak menaruh `aria-current` di DOM, jadi
aturan CSS `.…[aria-current="page"]` selalu gagal. Solusinya: `inject_css()` menerima halaman
aktif dan menyuntik aturan khusus untuk `href` halaman itu, ditentukan dari `st.context.url`
di `app.py`.

**e. Popover menu HP tidak ter-styling.** Panel popover dirender di portal dengan
`data-testid="stPopoverBody"`, bukan `div[data-baseweb="popover"]`. Selector diperbaiki
sehingga menu HP ikut berlatar kaca dan menyorot halaman aktif.

**f. Komponen baru.**
- `band_html()` (`src/ui.py`): batang visual rentang prediksi 80% dengan penanda nilai tengah
  di kartu hasil Prediksi — sebelumnya rentang hanya ditulis sebagai teks.
- `empty_state(icon, title, text)`: kotak kosong yang ramah, menggantikan `st.warning`
  di dashboard, Cek Harga, Prediksi, Bandingkan Wilayah, dan Tren.
- `page_header`/`section_title`/`hero` menerima `eyebrow` (label kecil di atas judul).
- Panel chart/tabel dibungkus `st.container(key="pc-card-…")`, filter dibungkus
  `key="pc-bar-…"`, sehingga otomatis tampil sebagai kartu kaca tanpa menulis HTML berulang.

**g. Poles rinci.**
- `style_fig()`: tooltip gabungan (`hovermode="x unified"`), grid titik halus, judul rata
  kiri, tipografi Plus Jakarta Sans, dan toolbar Plotly disembunyikan.
- Hero dipadatkan (semua breakpoint): tinggi 449 px → 331 px, `flex-wrap:nowrap` supaya strip
  ilustrasi tidak turun baris.
- Grid kartu responsif 4 → 3 → 2 → 1 kolom.
- Kontras diuji terprogram: `--muted-2` dari `#7E8B69` (3,55:1 — gagal WCAG AA) menjadi
  `#64734C` (5,0:1). Semua pasangan teks/latar lain ≥ 4,5:1.
- Legenda batang rentang dibuat satu baris dan dipendekkan jadi nominal saja
  (sebelumnya membungkus 2 baris di layar 360 px).
- `@media (prefers-reduced-motion: reduce)` mematikan animasi.

> Catatan: perbaikan (c) dan (d) **bukan** permintaan revamp, melainkan bug yang baru
> ketahuan setelah browser sungguhan dipakai untuk verifikasi. Sebelumnya verifikasi hanya
> memakai `AppTest` (struktural), yang tidak bisa menangkap CSS yang tidak pernah cocok.

---

### 3.14 Dokumentasi

`README.md` diperbaiki:
- Struktur folder: `src/ui/` → `src/ui.py`, `test_model_policy.py` ditambahkan,
  `requirements-dev.txt` dan file turunan didokumentasikan.
- Panduan Linux/macOS ditambahkan (sebelumnya hanya Windows).
- Bagian reproduksibilitas: menjelaskan mengapa sklearn/xgboost dipin persis.
- Bagian **Catatan Perbaikan Terakhir** berisi tabel ringkas semua perbaikan.
- Jumlah test disesuaikan (21/21 dan 43/43).

---

## 4. Test yang ditambahkan

Perbaikan disertai test regresi agar tidak kembali rusak diam-diam.

**`tests/test_all.py`** (+12 test, total 33):
- `filter Periode dengan satu tanggal`
- `ganti komoditas saat kabupaten terpilih`
- `detail komoditas direset saat pilihan menyusut`
- `inject_css` menerima parameter halaman aktif & menyuntik aturan `href`
- pemetaan slug halaman lengkap, Beranda memakai `href` kosong
- 8 pengecekan token CSS (`--up-bg`, `--down-bg`, `--muted-2`, `pc-band-dot`,
  `pc-empty`, `stPopoverBody`, `st-key-pc-card-`, `prefers-reduced-motion`)

**`tests/test_model_policy.py`** (+11 test, total 48):
- format `idr` ribuan / 1 desimal / 2 desimal / NaN
- format `fmt_pct` desimal koma
- `esc()` menetralkan tag HTML
- `band_html()`: 3 label, posisi titik, rentang nol, escape label, nilai di luar rentang

---

## 5. Hasil verifikasi akhir

| Pemeriksaan | Hasil |
|---|---|
| `python tests/test_all.py` | **33 / 33 lulus** |
| `python tests/test_model_policy.py` | **48 / 48 lulus** |
| `pyflakes app.py retrain.py src/*.py pages/*.py tests/*.py` | **bersih (exit 0)** |
| Notebook 01–05 dieksekusi ulang | **0 error** |
| `python retrain.py` penuh (salinan terpisah) | **berhasil** |
| Aplikasi dari clone bersih | **HTTP 200**, test lulus |
| Notebook 05 di clone bersih | **0 error**, verifikasi benar |
| Uji browser nyata (Chromium headless) | 6 halaman, 0 exception, 0 scroll horizontal |
| Kontras teks (WCAG AA) | semua pasangan ≥ 4,5:1 |

**Reproduksibilitas retrain.** Pelatihan deterministik (`SEED = 42`). Hasil retrain ulang di
lingkungan ini **tidak bit-identik** dengan model ter-commit (selisih MAPE ~0,001%) karena
versi numpy/pandas sedikit berbeda (2.5.5/3.1.1 vs 2.5.4/3.0.6 di metadata). Model terpilih,
metode per komoditas, dan kesimpulan **tidak berubah**.

---

## 6. Yang sengaja TIDAK diubah

Agar jelas batas pekerjaan ini:

| Hal | Alasan tidak diubah |
|---|---|
| Angka evaluasi & model | Tidak ada bug; mengubahnya akan memutus jejak hasil yang sudah dilaporkan |
| Metode per komoditas (6 naive + 1 model) | Ini keputusan metodologis yang sudah didokumentasi & benar |
| `prices_long.csv` dan `*.pkl` di Git | Dibutuhkan agar aplikasi jalan di Streamlit Cloud |
| Horizon prediksi 2–3 bulan | Sudah sengaja disembunyikan karena belum dievaluasi |
| Fitur `geo_id` | Sudah benar dibuang (urutan angka tidak bermakna) |
| Tanggal Ramadan 2016–2022 | Sudah dicatat di README sebagai belum diverifikasi |
| Git history (rewrite / LFS) | Berisiko dan di luar lingkup; sudah disarankan di README |

---

## 7. Cara memverifikasi ulang

```bash
cd PasarCerdas

# 1. Siapkan environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# 2. Jalankan test
python tests/test_all.py           # harapan: 21/21 lulus
python tests/test_model_policy.py  # harapan: 43/43 lulus

# 3. Jalankan aplikasi
python -m streamlit run app.py
```

Cek manual yang disarankan di browser:
1. **Cek Harga** → pilih satu tanggal di filter *Periode* → tabel harus ikut menyaring.
2. **Cek Harga** → pilih Telur → Nusa Tenggara Barat → Sumbawa → ganti bahan ke Beras →
   halaman tidak boleh error.
3. **Beranda** → angka harga harus berformat `Rp 28.640` (bukan `Rp 28,640`).
4. **Prediksi** → tekan tombol → kisaran 80% tampil dengan format Rupiah yang benar.

---

## 8. Commit

Semua perbaikan berada di atas basis `1a2ea52`. Lihat daftar lengkapnya dengan:

```bash
git log --oneline 1a2ea52..HEAD
```

Ringkasan isi tiap commit:

| Commit | Isi |
|---|---|
| `8035dcc` | Finalisasi: bug UI, format Rupiah, filter tanggal, prediksi, kebersihan repo |
| `b00d53b` | Bersihkan import tak terpakai di tests |
| `1bc4970` | README: sesuaikan jumlah pengujian dan cakupan test |
| `71d1980` | Dokumen ini (`CATATAN_PERBAIKAN.md`) |
| berikutnya | Revamp tampilan (bagian 3.15): design system, panel kartu, menu aktif, komponen baru |

Untuk melihat seluruh perubahan:

```bash
git diff 1a2ea52..HEAD
git show --stat 8035dcc
```

Untuk mengirim ke remote (setelah kamu review):

```bash
git push origin main
```

---

## 9. Saran lanjutan (belum dikerjakan)

Hal-hal yang aku temukan tapi **tidak** kukerjakan karena di luar lingkup "perbaiki dan
finalisasi", diurutkan dari yang paling berdampak:

1. **Backtest bergulir (rolling origin).** Validasi per komoditas hanya memakai 2024. Backtest
   2022→2026 akan jauh lebih meyakinkan daripada satu tahun tunggal.
2. **Perkecil ukuran model.** `best_model.pkl` 18 MB dan `models/alternatif/` 21 MB. Kompresi
   lebih agresif atau `max_features` lebih kecil bisa memangkas drastis tanpa kehilangan akurasi.
3. **Git LFS** untuk `*.pkl` dan `prices_long.csv` agar riwayat tidak membengkak tiap retrain.
4. **Cache `build_features`** di level data (bukan hanya Streamlit), karena masih dipakai
   di beberapa jalur dan butuh ~25 detik untuk seluruh dataset.
5. **Uji aksesibilitas & visual** dengan browser sungguhan. Di lingkungan ini tidak tersedia
   Chrome, jadi verifikasi UI memakai `AppTest` (struktural, bukan visual).
   *(Sudah sebagian dikerjakan: Chromium headless via Playwright dipasang sementara untuk revamp
   tampilan; temuan selector CSS mati dan menu aktif tidak tersorot berasal dari situ. Playwright
   tidak dimasukkan ke `requirements.txt`. Untuk mengulang:
   `pip install playwright && python -m playwright install chromium`.)*
6. **Verifikasi tanggal Ramadan/Idul Fitri 2016–2022** ke sumber resmi Kemenag.
