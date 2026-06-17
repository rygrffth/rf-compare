import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    dataset_filename = 'dataset.csv'
    output_folder = 'hasil_perbandingan_ml'
    
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        df = pd.read_csv(dataset_filename)
        print(f"Dataset '{dataset_filename}' berhasil dimuat dengan {len(df)} baris data.")
    except FileNotFoundError:
        print(f"Error: File '{dataset_filename}' tidak ditemukan.")
        return

    X = df[['mean_voltage', 'std_dev_voltage', 'mean_current', 'std_dev_current']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Data dibagi menjadi {len(X_train)} sampel latih dan {len(X_test)} sampel uji.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_config = {
        'Random Forest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                'n_estimators': [100, 200],
                'max_depth': [15, None],
                'min_samples_leaf': [1, 2]
            },
            False
        ),
        'Decision Tree': (
            DecisionTreeClassifier(random_state=42),
            {
                'criterion': ['gini', 'entropy'],
                'max_depth': [10, 20, None],
                'min_samples_leaf': [1, 2]
            },
            False
        ),
        'SVM (RBF)': (
            SVC(probability=True, random_state=42, cache_size=1000),
            {
                'C': [1, 10],
                'gamma': ['scale']
            },
            True
        ),
        'SVM (Linear)': (
            LinearSVC(random_state=42, max_iter=5000, dual=False),
            {
                'C': [0.1, 1.0]
            },
            True
        ),
        'Logistic Regression': (
            LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            {
                'C': [0.1, 1.0, 10.0]
            },
            True
        ),
        'MLP Neural Network': (
            MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
            {
                'hidden_layer_sizes': [(64, 32), (100,)],
                'alpha': [0.0001, 0.001]
            },
            True
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(random_state=42),
            {
                'n_estimators': [100],
                'learning_rate': [0.1],
                'max_depth': [3, 5]
            },
            False
        ),
        'AdaBoost': (
            AdaBoostClassifier(random_state=42),
            {
                'n_estimators': [50, 100],
                'learning_rate': [0.5, 1.0]
            },
            False
        ),
        'Naive Bayes (Gaussian)': (
            GaussianNB(),
            {},
            True
        ),
        'LDA': (
            LinearDiscriminantAnalysis(),
            {},
            True
        ),
        'QDA': (
            QuadraticDiscriminantAnalysis(),
            {},
            True
        )
    }

    all_metrics = {}
    target_names = ['Off Contact', 'Normal', 'Inisiasi Busur Api', 'Busur Api Konstan']

    print("\n==================================================")
    print("MEMULAI PROSES TRAINING & EVALUASI MODEL")
    print("==================================================")

    for model_name, (model, param_grid, use_scaled) in tqdm(models_config.items(), desc="Evaluasi Model"):
        print(f"\n>>> Melatih Model: {model_name}...")
        start_time = time.time()
        
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        
        best_params = None
        if param_grid:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1,
                scoring='accuracy'
            )
            grid_search.fit(X_tr, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Hyperparameter Terbaik: {best_params}")
        else:
            best_model = model
            best_model.fit(X_tr, y_train)
            print("Model dilatih langsung tanpa GridSearchCV.")
            
        training_time = time.time() - start_time
        
        y_pred = best_model.predict(X_te)
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        report_txt = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        
        all_metrics[model_name] = {
            'Akurasi': accuracy,
            'Presisi (W)': report_dict['weighted avg']['precision'],
            'Recall (W)': report_dict['weighted avg']['recall'],
            'F1-Score (W)': report_dict['weighted avg']['f1-score'],
            'Waktu Training (s)': training_time
        }
        
        model_subfolder = os.path.join(output_folder, model_name.replace(' ', '_').replace('(', '').replace(')', ''))
        os.makedirs(model_subfolder, exist_ok=True)
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names,
            annot_kws={"size": 12, "weight": "bold"}
        )
        plt.title(f'Confusion Matrix - {model_name} (Akurasi: {accuracy:.4f})', fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('Label Aktual', fontsize=12)
        plt.xlabel('Label Prediksi', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(model_subfolder, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        with open(os.path.join(model_subfolder, 'laporan_klasifikasi.txt'), 'w') as f:
            f.write(report_txt)
            
        with open(os.path.join(model_subfolder, 'detail_model.txt'), 'w') as f:
            f.write(f"Nama Model: {model_name}\n")
            f.write(f"Akurasi: {accuracy:.6f}\n")
            f.write(f"Presisi (Weighted): {report_dict['weighted avg']['precision']:.6f}\n")
            f.write(f"Recall (Weighted): {report_dict['weighted avg']['recall']:.6f}\n")
            f.write(f"F1-Score (Weighted): {report_dict['weighted avg']['f1-score']:.6f}\n")
            f.write(f"Waktu Training: {training_time:.2f} detik\n")
            if best_params:
                f.write(f"Hyperparameter Terbaik: {best_params}\n")
            else:
                f.write("Model dilatih menggunakan parameter default.\n")
                
        print(f"Hasil untuk {model_name} berhasil disimpan di folder: '{model_subfolder}'")

    comparison_df = pd.DataFrame(all_metrics).T.sort_values(by='Akurasi', ascending=False)
    comparison_df.to_csv(os.path.join(output_folder, 'tabel_perbandingan_model.csv'))
    
    print("\n\n==================== TABEL PERBANDINGAN MODEL ====================")
    print(comparison_df.round(4))
    print("==================================================================")

    print("\nMembuat grafik visualisasi performa...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(16, 9))
    metrics_to_plot = comparison_df[['Akurasi', 'Presisi (W)', 'Recall (W)', 'F1-Score (W)']]
    metrics_to_plot.plot(kind='bar', ax=ax, rot=45, width=0.85, colormap='viridis')
    plt.title('Perbandingan Metrik Performa Model Machine Learning', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Skor', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(max(0, comparison_df['Akurasi'].min() - 0.05), 1.01)
    plt.xticks(ha='right')
    plt.legend(title='Metrik', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8, rotation=90)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'grafik_perbandingan_performa.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    time_sorted = comparison_df['Waktu Training (s)'].sort_values()
    colors = sns.color_palette("rocket", len(time_sorted))
    ax_time = sns.barplot(x=time_sorted.values, y=time_sorted.index, hue=time_sorted.index, palette=colors, legend=False)
    plt.title('Perbandingan Waktu Pelatihan Model (Detik)', fontsize=14, fontweight='bold')
    plt.xlabel('Waktu Pelatihan (detik)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    
    for i, v in enumerate(time_sorted.values):
        ax_time.text(v + (time_sorted.values.max()*0.01), i, f'{v:.2f} s', va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'grafik_waktu_training.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SUKSES] Seluruh output berhasil disimpan di folder '{output_folder}/'")

if __name__ == '__main__':
    main()
