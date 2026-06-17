# Arc Detection Machine Learning Comparison

Repository ini berisi kode dan dataset untuk melakukan pengujian dan perbandingan 11 algoritma Machine Learning pada deteksi busur api. Pengujian dilakukan untuk melihat performa terbaik dari berbagai algoritma, dengan **Random Forest** berhasil menempati peringkat tertinggi baik dalam pengujian standar maupun pengujian anti-kebocoran data.

## Algoritma yang Dibandingkan

1. Random Forest
2. Decision Tree
3. Support Vector Machine - RBF Kernel
4. Support Vector Machine - Linear Kernel
5. Logistic Regression
6. Multi-Layer Perceptron Neural Network
7. Gradient Boosting
8. AdaBoost
9. Naive Bayes Gaussian
10. Linear Discriminant Analysis
11. Quadratic Discriminant Analysis

## Persyaratan

Sebelum menjalankan kodingan di repository ini, pastikan Anda telah melakukan langkah-langkah berikut:

### 1. Instalasi Python
Pastikan **Python versi 3.8 ke atas** sudah terinstal di komputer Anda.

### 2. Instalasi Library
Buka terminal lalu jalankan perintah berikut untuk menginstal library yang dibutuhkan:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

### 3. Persiapan Dataset
Pastikan file-file dataset berikut berada di dalam folder yang sama dengan script Python:
- `dataset.csv` — Dataset utama yang sudah diekstraksi fiturnya, digunakan untuk Skenario A
- `dataset_urut.csv` — Dataset dengan urutan kronologis, digunakan untuk Skenario B

## Cara Menjalankan

Terdapat dua skenario pengujian utama:

### Skenario A - Pengujian Standar
Skenario ini membagi data secara acak menggunakan fungsi train_test_split.
```bash
python perbandingan.py
```
Output berupa grafik performa, tabel perbandingan, dan laporan confusion matrix akan otomatis tersimpan di folder `hasil_perbandingan_ml/`.

### Skenario B - Pengujian Anti-Leakage
Skenario ini membagi data dengan memperhatikan urutan waktu secara kronologis dan melakukan augmentasi pada data training untuk memastikan tidak ada kebocoran data antara data latih dan data uji.
```bash
python perbandingan_anti_leakage.py
```
Output berupa grafik performa dan laporan klasifikasi akan tersimpan di folder `hasil_perbandingan_anti_leakage/`.

## Hasil Evaluasi

Metrik evaluasi yang digunakan meliputi:
- Akurasi
- Presisi Weighted
- Recall Weighted
- F1-Score Weighted
- Waktu Pelatihan

Hasil selengkapnya dapat dilihat langsung pada log yang tercetak di terminal, atau melalui laporan tabel berformat CSV dan gambar PNG yang tersimpan otomatis di dalam direktori output masing-masing skenario.
