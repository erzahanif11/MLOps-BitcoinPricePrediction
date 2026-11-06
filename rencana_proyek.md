# LK-01 : Proposal Project
## Machine Learning Operations (MLOps)
### Sistem Prediksi Harga Bitcoin Multi-Horizon Berbasis Data Historis dengan Continuous Training

**Dosen Pengampu:** Rizal Setya Perdana, S.Kom., M.Kom., Ph.D.

**Disusun oleh:**
Erza Hanif Pramudita Hanggara (245150200111038)

**FAKULTAS ILMU KOMPUTER**
**UNIVERSITAS BRAWIJAYA**
**2026**

---

## 1. Identifikasi Masalah

### 1.1 Domain

Proyek ini berada pada domain digital finansial, khususnya cryptocurrency dengan objek Bitcoin (BTC). Pasar cryptocurrency memiliki karakteristik yang sangat dinamis karena perdagangan berlangsung sepanjang waktu dan harga dapat mengalami perubahan dalam waktu yang relatif singkat. Kondisi tersebut menyebabkan pola yang dipelajari model dari data historis dapat berubah seiring waktu.

Model yang hanya dilatih satu kali berpotensi mengalami penurunan performa ketika menghadapi kondisi pasar terbaru. Oleh karena itu, dibutuhkan sistem prediksi yang tidak hanya menghasilkan prediksi, tetapi juga mampu menerima data baru secara berkala dan memperbarui model ketika diperlukan.

### 1.2 ML Task

Task yang digunakan pada proyek kali ini adalah regresi untuk time-series forecasting. Sistem akan memprediksi harga Bitcoin pada beberapa horizon waktu ke depan berdasarkan data pasar historis dari CoinGecko. Horizon merupakan rentang waktu ke depan yang menjadi cakupan prediksi suatu model berdasarkan data yang tersedia pada waktu prediksi. Multi-Horizon berarti model memiliki beberapa horizon (rentang waktu ke depan) yang dapat diprediksi.

Dalam penelitian ini, forecasting horizon digunakan untuk menentukan sejauh mana model memprediksi nilai pada periode mendatang. Sebagai contoh, apabila data historis digunakan hingga waktu (t) dan horizon ditetapkan selama 1 jam, maka model akan memprediksi nilai pada waktu (t+1) jam. Horizon prediksi yang direncanakan meliputi:

- 1 jam ke depan
- 3 jam ke depan
- 6 jam ke depan
- 12 jam ke depan
- 24 jam ke depan

Dengan demikian, pengguna dapat memilih horizon prediksi sesuai kebutuhan dengan alur sebagai berikut:

```
Data pasar sampai waktu t → horizon → model → prediksi harga pada t + horizon
```

Sistem menggunakan 1 model regresi. Nilai horizon digunakan sebagai salah satu fitur/input model sehingga tidak perlu model yang berbeda untuk setiap horizon.

---

## 2. Analisis Karakteristik Data Bergerak

### 2.1 Sumber Data

Sumber data yang digunakan adalah CoinGecko API. Data yang digunakan berfokus pada Bitcoin (BTC) dengan data yang terus bertambah, sesuai dengan kebutuhan proyek yang menggunakan data dinamis.

*Gambar 1. Halaman Harga Bitcoin pada API CoinGecko*

### 2.2 Karakteristik Data

Data yang digunakan memiliki karakteristik time-series, karena tiap observasi berhubungan dengan waktu tertentu. Data akan diambil dengan interval tiap jam (hourly) sesuai dengan forecasting yang memprediksi beberapa jam ke depan.

Data yang akan digunakan antara lain:

- **timestamp** → Unix epoch timestamp dalam satuan milidetik.
- **prices** → Menyimpan harga aset pada waktu tertentu.
- **market_caps** → Kapitalisasi pasar Bitcoin pada waktu t. (Market capitalization = harga × jumlah koin beredar)
- **total_volumes** → Total volume transaksi pada waktu t.

Struktur data historical market dapat direpresentasikan secara sederhana sebagai:

```json
{
  "prices": [
    [timestamp, price]
  ],
  "market_caps": [
    [timestamp, market_cap]
  ],
  "total_volumes": [
    [timestamp, volume]
  ]
}
```

*Gambar 2. Tes Endpoint API CoinGecko*

Tes endpoint dilakukan dengan mengakses endpoint API CoinGecko dengan parameter `days=30` dan `interval=hourly` yang akan mengambil data 30 hari ke belakang dengan interval waktu 1 jam per record.

### 2.3 Feature Engineering

Data dari CoinGecko akan diproses untuk membentuk fitur yang digunakan sebagai input model regresi. Fitur yang digunakan terdiri dari data pasar saat ini, fitur historis, dan fitur turunan.

Fitur yang direncanakan meliputi:

- **Price** → Harga Bitcoin pada waktu prediksi (t).
- **Market cap** → Kapitalisasi pasar Bitcoin pada waktu t.
- **Volume** → Total volume transaksi pada waktu t.
- **Lag** → Nilai suatu fitur pada beberapa periode sebelumnya, misalnya price lag-1, price lag-3, dan price lag-6.
- **Price Change / Return** → Perubahan harga antar periode yang merepresentasikan pergerakan relatif harga.
- **Volatility** → Tingkat perubahan atau fluktuasi harga dalam periode tertentu.
- **Horizon** → Jarak waktu prediksi yang digunakan sebagai input model, yaitu 1, 3, 6, 12, atau 24 jam.

Target yang diprediksi adalah harga Bitcoin pada waktu t + horizon, sedangkan fitur input menggunakan informasi yang tersedia hingga waktu t sehingga tidak menggunakan informasi dari masa depan, sesuai dengan fungsi dari forecasting itu sendiri.

---

## 3. Definisi Strategi Continual Learning

### 3.1 Strategi Pembaruan Model

Proyek menggunakan pendekatan Continuous Training (CT). Data terbaru akan diambil secara berkala dari CoinGecko API dan ditambahkan ke dataset. Namun model tidak selalu dilatih ulang setiap kali prediksi dilakukan. Retraining dilakukan apabila terdapat kondisi yang menjadi trigger, seperti data drift dan penurunan performa model pada tahap monitoring.

### 3.2 Data Ingestion Berkelanjutan

Data terbaru akan diperoleh dari CoinGecko API secara berkala dan kemudian diproses dalam resolusi tiap jam (hourly) agar konsisten dengan model forecasting.

```
CoinGecko API → Ingestion berkala → Validasi → Storage
```

### 3.3 Monitoring Data Drift & Keputusan Retraining

Monitoring dilakukan dengan memantau karakteristik data terbaru dan dibandingkan dengan distribusi data referensi yang digunakan untuk melatih model terakhir. Monitoring ini dilakukan secara berkelanjutan pada fitur-fitur yang merepresentasikan kondisi pasar dan digunakan oleh model, sehingga perubahan distribusi fitur dapat menjadi indikasi data drift.

Beberapa fitur yang dipantau yakni:

- **Price** → Harga Bitcoin pada waktu prediksi (t).
- **Market cap** → Kapitalisasi pasar Bitcoin pada waktu t.
- **Volume** → Total volume transaksi pada waktu t.
- **Price change / Return** → Perubahan harga antar periode yang merepresentasikan pergerakan relatif harga.
- **Volatility** → Tingkat perubahan atau fluktuasi harga dalam periode tertentu.

Price, market cap, dan volume merupakan data utama dari sumber API, sedangkan return dan volatility merupakan fitur turunan. Selain data drift, monitoring juga dilakukan terhadap performa model menggunakan data aktual yang didapatkan kemudian menggunakan metrik evaluasi.

Retraining akan dipicu (trigger) apabila:

1. Data drift melewati threshold, **atau**
2. Performa model mengalami penurunan melewati threshold.

```
Data Baru → Monitoring → Trigger Check → Retraining
```

### 3.4 Proses Retraining

Ketika salah satu kondisi trigger terpenuhi, sistem menjalankan proses retraining untuk memperbarui model agar dapat menyesuaikan diri dengan kondisi data terbaru. Proses retraining kurang lebih sama dengan proses training model di awal, namun menggunakan data training yang diperbarui dengan data terbaru melalui mekanisme full retraining (menambahkan data baru pada data lama).

Karena sistem menggunakan satu model regresi untuk beberapa horizon, dataset training mencakup data untuk horizon 1, 3, 6, 12, dan 24 jam dengan nilai horizon sebagai salah satu fitur input. Model hasil retraining kemudian dievaluasi menggunakan metrik yang telah ditentukan. Apabila memenuhi kriteria performa, model dicatat sebagai versi baru pada Model Registry untuk menggantikan model sebelumnya.

Alur continuous training:

```
Data Historis Terbaru → Data Processing → Feature Engineering → Pembentukan Horizon → Training → Evaluation → Model Baru (Registry) → Deployment
```

---

## 4. Penetapan Kriteria Keberhasilan

### 4.1 Metrik Teknis

Karena Machine Learning Task yang digunakan adalah regresi forecasting, maka metrik utama yang digunakan adalah:

- **MAE (Mean Absolute Error)**: Mengukur rata-rata selisih absolut antara harga aktual dan harga hasil prediksi.
- **RMSE (Root Mean Squared Error)**: Mengukur error prediksi dengan memberikan penalti lebih besar terhadap error yang besar.
- **MAPE (Mean Absolute Percentage Error)**: Mengukur kesalahan prediksi dalam bentuk persentase.

Semua metrik yang digunakan (MAE/RMSE/MAPE) akan dievaluasi per horizon (1h, 3h, 6h, 12h, 24h) untuk mengetahui kemampuan model dalam memprediksi berbagai horizon.

Selain performa forecasting, sistem juga akan memperhatikan metrik production seperti:

- **Inference latency** → Waktu yang diperlukan model untuk menghasilkan prediksi;
- **Data freshness** → Seberapa baru data yang digunakan oleh sistem;
- Keberhasilan data ingestion;
- Keberhasilan pipeline training.

### 4.2 Metrik Bisnis

Metrik bisnis yang digunakan lebih berfokus pada manfaat operasional sistem, yaitu:

- Efisiensi proses analisis harga;
- Waktu yang dibutuhkan untuk memperoleh prediksi;
- Kemampuan sistem menghasilkan prediksi secara otomatis;
- Kemampuan model beradaptasi terhadap perubahan kondisi pasar.

ROI dapat digunakan sebagai evaluasi tambahan apabila dilakukan simulasi penggunaan hasil prediksi, tetapi bukan merupakan tujuan utama proyek.

---

## 5. Perancangan Diagram Arsitektur Pipeline

### 5.1 Rancangan Pipeline

Arsitektur sistem dirancang untuk mencakup proses data ingestion, version control, model registry, inference, monitoring, dan continuous training.

*Gambar 3. Diagram Pipeline Sistem*

Alur sistem dimulai dari data ingestion melalui API CoinGecko yang dilakukan secara berkala. Data yang diperoleh kemudian melalui tahap data validation untuk memastikan data sesuai dengan format dan kebutuhan sistem. Selanjutnya, dilakukan data versioning sebagai checkpoint untuk menyimpan versi dataset yang digunakan. Setelah tahap ingestion, dilakukan preprocessing, yaitu pengolahan data dan penyelarasan data ke dalam resolusi hourly (1 jam). Kemudian dilakukan feature engineering untuk membentuk fitur-fitur yang relevan sebagai input model, seperti harga, volume, lag, return, dan horizon.

Setelah preprocessing selesai, dilakukan model training dengan menambahkan horizon sebagai salah satu fitur input. Dengan demikian, satu model dapat mempelajari hubungan antara kondisi pasar dan jarak waktu prediksi, sehingga dapat digunakan untuk memprediksi harga pada beberapa horizon yang telah ditentukan. Model hasil training kemudian dievaluasi menggunakan metrik yang telah ditetapkan. Jika model memenuhi kriteria performa, model tersebut dicatat dan dikelola dalam Model Registry menggunakan MLflow sebagai versi model yang siap digunakan.

Tahap selanjutnya adalah inference, yaitu penggunaan model production untuk menghasilkan prediksi harga berdasarkan data terbaru dan horizon yang dipilih. Selama model digunakan, sistem melakukan monitoring secara berkelanjutan untuk mendeteksi data drift maupun penurunan performa. Jika kondisi yang telah ditentukan sebagai trigger terpenuhi, proses retraining akan dijalankan untuk menghasilkan model yang lebih sesuai dengan kondisi data terbaru.

### 5.2 Alur Inferensi

Pada saat digunakan, sistem mengambil data pasar terbaru dan menerima nilai horizon yang dipilih oleh pengguna.

```
Data Terbaru + horizon → Production model → Prediksi harga x jam ke depan
```

Jika pengguna memilih horizon 24, maka x = 24:

```
Data Terbaru + horizon = 24 → Production model → Prediksi harga 24 jam ke depan
```