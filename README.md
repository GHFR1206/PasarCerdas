# PasarCerdas — Indonesia Food Price Intelligence

**Pengembangan Sistem Prediksi dan Analisis Harga Komoditas Pangan di Indonesia Menggunakan Machine Learning**

<p align="center">
  <a href="https://pasarcerdas.streamlit.app/">
    <strong>🌐 Buka Aplikasi PasarCerdas</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/ctrl-stiwww/PasarCerdas/">
    <strong>GitHub Repository</strong>
  </a>
</p>

PasarCerdas merupakan aplikasi web berbasis Streamlit untuk memantau, membandingkan, menganalisis, dan memprediksi harga komoditas pangan di Indonesia. Aplikasi ini menggunakan data historis World Bank *Monthly Food Price Estimates by Product and Market – Indonesia* (RTFP).

Sistem mencakup **7 komoditas pangan** (telur, bawang putih, daging ayam, minyak nabati, bawang merah, beras, dan gula) yang tersebar di **222 pasar** dalam dataset.

---

## 1. Keputusan Metodologis

Beberapa keputusan penting dalam pengolahan dataset:

| Temuan pada Dataset | Keputusan |
|---|---|
| Kolom tanpa prefiks (`rice`, `eggs`, dan lainnya) merupakan harga observasi, sedangkan `c_*` merupakan *close estimate* dari model World Bank. | Target dan riwayat harga menggunakan harga observasi. `c_*` hanya digunakan sebagai referensi. |
| Dataset memiliki 223 `geo_id`, terdiri dari 222 pasar dan 1 agregat nasional (*Market Average*). | Baris agregat nasional dikeluarkan dari analisis dan pemodelan. |
| Observasi pasar tersedia sejak 2016, tetapi data lebih padat pada Januari 2020–September 2026. | Pemodelan menggunakan data observasi yang memenuhi kriteria kelayakan. |
| Oktober–November 2026 belum memiliki observasi aktual. | Prediksi website ditujukan untuk Oktober 2026. |
| Terdapat 30 observasi dengan rasio terhadap estimasi di luar rentang 0,5×–2×. | Data ditandai sebagai `is_outlier` dan dikeluarkan dari pemodelan, tetapi tidak dihapus dari dataset. |
| `spatially_interpolated` bernilai 0 pada seluruh baris. | Tidak ada interpolasi spasial yang perlu dipisahkan. |

**Kriteria data untuk pemodelan:**
- Target harga merupakan harga observasi.
- Harga bulan sebelumnya juga harus berupa observasi.
- Minimal terdapat 3 observasi dalam 6 bulan terakhir.

**Pembagian data berdasarkan waktu:**

| Dataset | Periode |
|---|---|
| Training | Hingga Desember 2023 |
| Validation | Januari–Desember 2024 |
| Testing | Januari 2025–September 2026 |
| Latih ulang model final | Hingga Desember 2024 (train + validation) |

Seleksi model dan hyperparameter hanya menggunakan data validasi (metrik MAPE) untuk menghindari penggunaan data testing dalam proses pemilihan model.

**Pilihan per komoditas.** Untuk setiap bahan, model terpilih dibandingkan dengan tebakan naif (harga bulan lalu) pada data validasi. Yang memiliki MAPE lebih kecil yang dipakai. Pada versi ini, 6 dari 7 bahan memakai harga bulan lalu, dan hanya bawang merah yang memakai model. Pilihan ini tidak memakai data test.

**Model final dilatih ulang** dengan data train + validation (sampai Desember 2024). Metrik test di metadata tetap berasal dari model yang hanya dilatih sampai Desember 2023, sehingga tetap out-of-sample.

**Fitur musiman:** penanda Ramadan (proporsi hari puasa dalam bulan), bulan Idul Fitri, dan bulan sebelum Idul Fitri, serta posisi bulan dalam setahun (sin/cos). Fitur ini hanya berisi informasi kalender. Tanggal Ramadan dan Idul Fitri per tahun ada di `src/feature_engineering.py`. Tanggal 2023–2026 sudah dicocokkan dengan penetapan pemerintah (Kemenag) lewat pemberitaan. Tanggal 2016–2022 belum diverifikasi.

**Fitur `geo_id` dibuang**, karena urutan angkanya tidak bermakna. Provinsi (`adm1_name`) di-one-hot.

*Catatan: Interpretasi kolom diverifikasi melalui katalog data World Bank dan pemeriksaan empiris.*

---

## 2. Fitur Website

PasarCerdas menyediakan beberapa halaman utama dengan navigasi yang dirancang agar mudah digunakan oleh masyarakat umum.

| Halaman | Fitur |
|---|---|
| Beranda | Ringkasan informasi dan kondisi harga pangan. |
| Cek Harga | Melihat harga berdasarkan provinsi, kabupaten/kota, pasar, dan periode. |
| Prediksi | Memperkirakan harga komoditas pada periode berikutnya. |
| Bandingkan Wilayah | Membandingkan harga antarwilayah, termasuk melalui peta. |
| Tren | Menganalisis perkembangan dan pola perubahan harga. |
| Seberapa Akurat? | Melihat hasil evaluasi dan performa model Machine Learning. |

Aplikasi menggunakan tampilan interaktif dengan palet warna hijau–biru, visualisasi data, kartu informasi, serta penjelasan untuk membantu pengguna memahami hasil yang ditampilkan.

---

## 3. Struktur Folder

```text
PasarCerdas/
│
├── app.py
├── run.bat
├── retrain.py
├── requirements.txt
├── requirements-dev.txt
│
├── assets/
│   └── style.css
│
├── dataset/
│   ├── raw/
│   │   └── IDN_RTFP_mkt_2007_2026-11-02.zip
│   └── processed/
│       ├── prices_long.csv            (dipakai aplikasi, ikut di-commit)
│       └── features_model.csv         (turunan; dibuat ulang oleh retrain.py, tidak di-commit)
│
├── notebooks/
│   ├── 01_dataset_audit_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── training.py
│   ├── evaluation.py
│   ├── prediction.py
│   ├── eda.py
│   ├── pipeline.py
│   └── ui.py
│
├── pages/
│   ├── dashboard.py
│   ├── harga_pangan.py
│   ├── prediksi.py
│   ├── perbandingan.py
│   ├── analisis_tren.py
│   └── evaluasi_model.py
│
├── models/
│   ├── best_model.pkl
│   ├── preprocessing.pkl
│   ├── model_metadata.json
│   └── alternatif/
│
├── results/
│   ├── evaluation_*.csv
│   ├── commodity_method.csv
│   ├── prediction_results.csv        (turunan; tidak di-commit)
│   ├── eda_summary.json
│   └── figures/
│
└── tests/
    ├── test_all.py
    └── test_model_policy.py
```

---

## 4. Persyaratan Sistem

| Komponen | Persyaratan |
|---|---|
| Sistem Operasi | Windows 10/11 64-bit (Linux/macOS juga didukung bila menjalankan manual) |
| Python | 3.11–3.13 |
| RAM | Minimal 4 GB |
| Koneksi Internet | Dibutuhkan saat instalasi pertama, serta untuk font Google dan peta OpenStreetMap. |

---

## 5. Cara Menjalankan Aplikasi

### A. Instalasi

1. Download dan instal Python melalui [python.org](https://www.python.org/downloads/).
2. Saat instalasi, centang opsi **Add python.exe to PATH**.
3. Download repository melalui GitHub atau gunakan tombol **Code → Download ZIP**.
4. Ekstrak file ZIP secara penuh.
5. Buka folder `PasarCerdas`.
6. Jalankan aplikasi dengan klik dua kali file `run.bat`.

Pada proses pertama, sistem akan membuat virtual environment `.venv` dan menginstal seluruh dependensi yang diperlukan. Proses ini membutuhkan koneksi internet.

### B. Membuka Aplikasi

Setelah proses instalasi selesai, Streamlit akan berjalan secara otomatis.

Buka alamat berikut melalui browser:

```text
http://localhost:8501
```

Untuk menghentikan aplikasi, tekan `Ctrl + C` pada jendela CMD atau tutup jendela tersebut.

### C. Menjalankan Secara Manual melalui CMD

```bat
cd path\ke\PasarCerdas

py -3 -m venv .venv

.venv\Scripts\activate

python -m pip install -r requirements.txt

python -m streamlit run app.py
```

### D. Menjalankan di Linux/macOS

`run.bat` khusus Windows. Di Linux/macOS gunakan perintah berikut:

```bash
cd path/ke/PasarCerdas
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 6. Menjalankan Notebook

Notebook digunakan untuk melihat tahapan pengolahan data, eksplorasi, pelatihan, dan evaluasi model.

Aktifkan virtual environment:

```bat
.venv\Scripts\activate
```

Pasang dependensi notebook (sekali saja):

```bat
python -m pip install -r requirements-dev.txt
```

Jalankan Jupyter Notebook:

```bat
jupyter notebook notebooks
```

Jalankan notebook secara berurutan:

1. `01_dataset_audit_eda.ipynb`
2. `02_preprocessing.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_model_training.ipynb`
5. `05_model_evaluation.ipynb`

Notebook 04 menjalankan eksperimen pelatihan ulang model, termasuk tuning hyperparameter, tanpa menimpa file model pada folder `models/`.

---

## 7. Melatih Ulang Model

Untuk menjalankan kembali seluruh pipeline Machine Learning:

```bat
.venv\Scripts\activate

python retrain.py
```

Pipeline mencakup:

1. Preprocessing data.
2. Exploratory Data Analysis (EDA).
3. Feature Engineering.
4. Pelatihan 12 eksperimen model beserta tuning hyperparameter.
5. Evaluasi model (MAPE, MAE, RMSE, R²) dan pemilihan metode per komoditas.
6. Perhitungan rentang ketidakpastian 80% per komoditas dan cek cakupannya pada data test.
7. Latih ulang model final dengan data train + validation.
8. Penyimpanan model dan metadata.

Untuk menjalankan pengujian cepat tanpa tuning:

```bat
python retrain.py --quick
```

**Catatan:** Mode `--quick` hanya digunakan untuk pengujian awal, bukan untuk menghasilkan hasil evaluasi akhir.

**Reproduksibilitas:** pelatihan bersifat deterministik (`SEED = 42`) pada versi pustaka yang sama. `scikit-learn` dan `xgboost` dipin persis di `requirements.txt` karena file `.pkl` dibuat dengan versi tersebut. Bila `numpy`/`pandas` berbeda patch version, angka evaluasi dapat bergeser sangat kecil (jauh di bawah 0,1%) tanpa mengubah model terpilih.

---

## 8. Mengganti Dataset

Untuk menggunakan dataset baru:

1. Letakkan file ZIP dataset di folder `dataset/raw/`.
2. Sesuaikan konfigurasi `RAW_ZIP` dan `RAW_CSV_NAME` pada `src/config.py`.
3. Pastikan format kolom dataset sesuai dengan RTFP Indonesia.
4. Sesuaikan periode training, validation, dan testing jika rentang data berubah.
5. Jalankan kembali pipeline:

```bat
python retrain.py
```

---

## 9. Troubleshooting

| Masalah | Solusi |
|---|---|
| Python belum terinstal | Instal Python 3.11–3.13 dan aktifkan opsi Add to PATH. |
| Versi Python tidak didukung | Gunakan Python 3.12.x |
| Instalasi dependensi gagal | Periksa koneksi internet atau proxy. Hapus folder `.venv`, lalu jalankan `run.bat` kembali. |
| Berkas penting tidak ditemukan | Pastikan ZIP sudah diekstrak secara penuh. |
| Port 8501 sedang digunakan | Jalankan Streamlit menggunakan port alternatif. |
| Peta atau font tidak tampil | Pastikan perangkat terhubung ke internet. |
| Error saat memuat model | Pastikan versi library sesuai dengan `requirements.txt`, atau jalankan ulang training. |

Untuk menggunakan port alternatif:

```bat
python -m streamlit run app.py --server.port 8502
```

---

## 10. Keterbatasan Sistem

- Data observasi aktual memiliki keterbatasan cakupan waktu dan lebih padat mulai sekitar tahun 2020.
- Model dilatih menggunakan data historis yang memenuhi kriteria kelayakan.
- Prediksi hanya tersedia untuk satu bulan ke depan. Horizon 2–3 bulan belum dievaluasi, sehingga tidak ditampilkan di aplikasi. Kode untuk itu masih ada di `predict_path`.
- Prediksi hanya tersedia untuk pasar yang memiliki observasi pada periode terakhir data.
- Validasi untuk memilih metode per komoditas hanya memakai satu tahun (2024). Stabilitas antar tahun ditampilkan di `commodity_method.csv` (kolom `mape_<tahun>`), tetapi belum ada backtest bergulir penuh.
- Rentang 80% dihitung dari error relatif pada data validasi. Ini bukan jaminan statistik, dan cakupannya pada data uji ditampilkan di aplikasi.
- Faktor di luar data harga (harga BBM, kurs, cuaca, panen) belum dimasukkan sebagai fitur.
- Performa model berbeda pada setiap komoditas. Naive Forecast masih menjadi baseline yang kuat untuk beberapa komoditas dengan pola harga relatif stabil.
- Model tidak dilatih ulang otomatis. Untuk memperbarui, jalankan `python retrain.py` setelah data baru tersedia.
- Data merupakan estimasi harga berdasarkan dataset World Bank, bukan harga resmi real-time seluruh pasar Indonesia.

---

## 11. Pengujian Sistem

Untuk menjalankan pengujian:

```bat
python tests/test_all.py
```

Pengujian mencakup:
- Import modul.
- Pemuatan model.
- Validasi hasil prediksi agar tidak menghasilkan nilai NaN.
- Pemeriksaan halaman aplikasi dan filter.
- Regresi filter bertingkat: ganti komoditas saat kabupaten/pasar sudah dipilih.
- Regresi filter **Periode** dengan satu tanggal.

Pengujian kebijakan model (`tests/test_model_policy.py`) memeriksa:
- Format Rupiah/persen dan escape HTML.
- `geo_id` tidak dipakai sebagai fitur dan fitur musiman tersedia.
- Perhitungan fitur Ramadan dan Idul Fitri.
- Metadata, CSV hasil, dan bundle model konsisten.
- Komoditas dengan metode naif memang memprediksi harga terakhir.
- Rentang ketidakpastian valid.

```bat
python tests/test_model_policy.py
```

---

## 12. Hasil Evaluasi Model

Hasil evaluasi dapat dilihat melalui halaman **Seberapa Akurat?** pada aplikasi atau file berikut:

```text
results/evaluation_metrics.csv
results/training_log.txt
```

Evaluasi dilakukan menggunakan metrik:
- **MAE (Mean Absolute Error):** mengukur rata-rata kesalahan absolut prediksi.
- **RMSE (Root Mean Squared Error):** memberikan penalti lebih besar pada kesalahan prediksi yang besar.
- **MAPE (Mean Absolute Percentage Error):** mengukur rata-rata kesalahan dalam persentase.
- **R² (R-squared):** mengukur kemampuan model dalam menjelaskan variasi target.

**Catatan membaca angka:** MAE dalam Rupiah didominasi oleh bahan yang harganya tinggi, sehingga untuk membandingkan antar bahan gunakan **MAPE per komoditas** (ditampilkan di halaman *Seberapa Akurat?*). Di halaman itu juga ditampilkan rentang 80% dan cakupannya pada data uji (idealnya mendekati 80%).

Hasil test gabungan untuk tiap metode (MAPE, Januari 2025–September 2026): model saja 5,18%, harga bulan lalu saja 4,91%, dan gabungan yang dipakai aplikasi **4,64%**.

**Cakupan rentang 80% di data uji** (sekitar 64–84%): untuk telur, bawang putih, dan daging ayam rentangnya terlalu sempit, karena harga pada 2025–2026 bergerak lebih besar dari pada validasi 2024. Untuk bahan lain cakupannya mendekati 80%.

---

## 13. Catatan Ukuran Repo

File `dataset/processed/prices_long.csv` (sekitar 58 MB) dan file model `.pkl` ikut di repo karena dibutuhkan saat aplikasi dijalankan di Streamlit Community Cloud. Setiap kali model dilatih ulang dan di-commit, riwayat Git bertambah sebesar file-file itu.

File **turunan** tidak lagi di-commit karena selalu dibuat ulang oleh `python retrain.py`:
- `dataset/processed/features_model.csv` (sekitar 18 MB)
- `results/prediction_results.csv` (sekitar 5 MB)

Keduanya masuk `.gitignore`. Notebook 05 sudah disesuaikan sehingga tidak lagi membaca `prediction_results.csv`, jadi notebook tetap bisa dijalankan setelah clone baru.

Opsi yang bisa dipertimbangkan:
- **Git LFS** untuk `*.pkl` dan `dataset/processed/*.csv`. Perlu di-install di komputer yang melakukan commit, dan riwayat lama tetap menyimpan file asli.
- **Commit model hanya saat model benar-benar berubah**, bukan setiap kali eksperimen.

## 14. Catatan Perbaikan Terakhir

Ringkasan perbaikan pada finalisasi ini:

| Bagian | Masalah | Perbaikan |
|---|---|---|
| `src/ui.py` | `idr()` memakai `replace(",", ".")` sehingga angka 1 desimal salah tampil (`Rp 1.234.6`) | Format Rupiah kini titik ribuan + koma desimal (`Rp 1.234,6`); `fmt_pct`/`idr` aman untuk NaN |
| `src/ui.py` | Nama pasar/provinsi disisipkan ke HTML tanpa escape | Ditambahkan `esc()` dan dipakai di komponen kartu |
| `src/ui.py` | Kartu sorotan menulis "paling turun" walau semua harga naik | Kalimat menyesuaikan kondisi data |
| `pages/dashboard.py` | Insight menyebut "paling turun" secara keliru; halaman error bila sebuah komoditas tidak punya pasangan harga di dua bulan | Insight mengikuti kondisi nyata; komoditas tanpa data dilewati dengan aman |
| `pages/harga_pangan.py` | Filter **Periode** mengabaikan input satu tanggal (diam-diam menampilkan semua data) | Satu tanggal kini difilter dengan benar; input kosong memakai rentang penuh |
| `pages/harga_pangan.py`, `pages/prediksi.py`, `pages/analisis_tren.py` | `selectbox` bertingkat bisa error ketika pilihan lama tidak ada lagi di opsi baru (mis. ganti komoditas setelah memilih kabupaten) | Nilai yang tidak valid direset ke pilihan pertama sebelum widget dirender |
| `pages/prediksi.py` | Label model dari `session_state` bisa tidak ada setelah retrain | Divalidasi dan fallback ke model pertama |
| `pages/perbandingan.py` | `.title()` mengubah nama resmi (`KOTA BAU-BAU` → `Kota Bau-Bau`, `DKI JAKARTA` → `Dki Jakarta`); judul tabel hardcode "10" walau slider bisa 5–20 | Nama ditampilkan apa adanya; judul mengikuti jumlah yang dipilih |
| `pages/evaluasi_model.py` | Klaim "keunggulan terlihat pada komoditas bawang merah" ditulis mati, tidak mengikuti data | Klaim dihitung dari `commodity_method.csv` |
| `src/prediction.py` | `predict_path` melempar `KeyError`/error tak tertangani untuk pasar tanpa observasi di bulan terakhir | Mengembalikan `InsufficientHistory` dengan pesan jelas |
| `src/prediction.py` | `eligible_markets` membangun seluruh tabel fitur (25 detik) untuk 1.483 pasangan pasar–komoditas | Dihitung langsung dari data observasi; hasil identik, jauh lebih cepat |
| `notebooks/05_model_evaluation.ipynb` | Sel verifikasi membandingkan prediksi model **refit** dengan `prediction_results.csv` dari model **pre-refit**, sehingga selalu mencetak `Konsisten: False` | Sel diperbaiki: memverifikasi metrik test metadata dapat direproduksi (selisih MAPE 0,001%) dan memeriksa prediksi refit valid |
| `.gitignore` | `features_model.csv` sudah di-ignore tetapi masih ter-track; `prediction_results.csv` ikut ter-commit | Keduanya dikeluarkan dari Git |
| `README.md` | Struktur folder menyebut `src/ui/` (sebenarnya `src/ui.py`), `test_model_policy.py` tidak didaftar, panduan hanya Windows | Struktur dan panduan diperbaiki, ditambah bagian Linux/macOS |
| Berbagai modul | Import dan variabel tak terpakai (`OrdinalEncoder`, `ID_COLS`, `metrics_table`, dll.) | Dibersihkan; `pyflakes` bersih |
| `assets/style.css` | Selector BaseWeb (`div[data-baseweb="select"]`, `button[data-baseweb="tab"]`, `div[data-testid="stPills"]`) tidak pernah cocok di Streamlit 1.67 — kustomisasi input dan tab tidak berefek | Diganti selector react-aria yang benar (`[data-testid="stSelectbox"] [role="group"]`, `[data-testid="stTab"]`, `[data-testid="stButtonGroup"]`, `[role="listbox"]`) |
| `app.py`, `src/ui.py` | Menu aktif tidak pernah tersorot karena Streamlit tidak menaruh `aria-current` di DOM | `inject_css()` menerima halaman aktif dari `st.context.url` dan menyuntik aturan `href` untuk halaman itu |
| `assets/style.css` | Popover menu HP tidak ter-styling (`div[data-baseweb="popover"]` tidak ada) | Diganti `[data-testid="stPopoverBody"]` |
| `src/ui.py`, `pages/*` | Tidak ada komponen untuk kondisi kosong, panel chart, dan rentang prediksi masih berupa teks | Ditambahkan `empty_state()`, panel `st.container(key="pc-card-…")`, dan `band_html()` (batang rentang 80%) |
| `assets/style.css` | `--muted-2` `#7E8B69` hanya 3,55:1 (gagal WCAG AA); hero terlalu tinggi karena strip ilustrasi turun baris | Warna digelapkan ke `#64734C` (5,0:1); hero dipadatkan 449 px → 331 px; ditambah `prefers-reduced-motion` |

Seluruh pengujian (`tests/test_all.py` 33/33 dan `tests/test_model_policy.py` 48/48) lulus, dan kelima notebook berjalan tanpa error.

## 15. Sumber Data

**World Bank — Monthly Food Price Estimates by Product and Market (Indonesia)**

[World Bank Microdata Library](https://microdata.worldbank.org/catalog/6166/get-microdata)

Dataset digunakan sebagai sumber data historis harga komoditas pangan untuk proses eksplorasi, pengolahan, pelatihan, dan evaluasi model Machine Learning.

---

<p align="center">
  <strong>PasarCerdas — Indonesia Food Price Intelligence</strong><br>
  Mengolah data historis menjadi informasi harga pangan yang lebih mudah dipahami.
</p>
