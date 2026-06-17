# Arc Detection Machine Learning Comparison

Repository ini berisi kode dan dataset untuk melakukan pengujian dan perbandingan 11 algoritma Machine Learning pada deteksi busur api (Arc Detection). Pengujian dilakukan untuk melihat performa terbaik dari berbagai algoritma, dengan **Random Forest** berhasil menempati peringkat tertinggi baik dalam pengujian standar maupun pengujian anti-kebocoran data (*anti-leakage*).

## Algoritma yang Dibandingkan (11 Model)
1. Random Forest
2. Linear Discriminant Analysis (LDA)
3. Support Vector Machine (SVM) - RBF Kernel
4. Multi-Layer Perceptron (MLP) Neural Network
5. Logistic Regression
6. Decision Tree
7. Gradient Boosting
8. Support Vector Machine (SVM) - Linear Kernel
9. Quadratic Discriminant Analysis (QDA)
10. Naive Bayes (Gaussian)
11. AdaBoost

## Persyaratan (Prerequisites)
Sebelum menjalankan kodingan di repository ini, pastikan Anda telah melakukan langkah-langkah berikut:

### 1. Instalasi Python
Pastikan bahasa pemrograman **Python (versi 3.8 ke atas)** sudah terinstal di komputer/laptop Anda.

### 2. Instalasi Library (Dependencies)
Anda perlu menginstal beberapa library Python yang digunakan dalam script ini. Buka terminal (Command Prompt / PowerShell) lalu jalankan perintah berikut:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

### 3. Persiapan Dataset
Kodingan ini membutuhkan dataset berformat `.csv`. Pastikan kelima file dataset mentah ini berada **di dalam folder yang sama** dengan script Python (`.py`):
- `datasetoffcontact.csv`
- `datanormal2000.csv`
- `dataarc_transient.csv`
- `dataarc_konstan.csv`
- `dataset.csv` (Digunakan khusus untuk Skenario A)

## Cara Menjalankan Kodingan

Terdapat dua skenario pengujian utama:

### A. Skenario Standar (Random Split)
Skenario ini membagi data secara acak menggunakan fungsi `train_test_split`.
Jalankan perintah berikut di terminal:
```bash
python perbandingan.py
```
Output, grafik performa, tabel perbandingan, dan laporan *confusion matrix* akan otomatis tersimpan di dalam folder `hasil_perbandingan_ml/`.

### B. Skenario Time-Series Anti-Leakage (Chronological Split)
Skenario ini membagi data dengan memperhatikan urutan waktu nyata (*chronological*) dan melakukan augmentasi secara ketat pada data *training* untuk memastikan tidak ada *data leakage*.
Jalankan perintah berikut di terminal:
```bash
python perbandingan_anti_leakage.py
```
Output, grafik performa, dan laporan klasifikasi akan secara otomatis tersimpan di dalam folder `hasil_perbandingan_anti_leakage/`.

## Hasil Evaluasi
Seluruh hasil pengujian metrik meliputi:
- **Akurasi**
- **Presisi (Weighted)**
- **Recall (Weighted)**
- **F1-Score (Weighted)**
- **Waktu Pelatihan (Training Time)**

Hasil selengkapnya dapat dilihat langsung pada log yang tercetak di layar terminal Anda, atau melalui laporan tabel berformat CSV dan gambar `.png` yang ter-generate otomatis di dalam direktori output masing-masing skenario.
