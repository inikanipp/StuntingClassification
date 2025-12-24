import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import boxcox
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Stunting Classification", layout="wide")

# Fungsi untuk memanggil file CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Panggil file CSS
local_css("style.css")

# --- JUDUL UTAMA  ---
st.markdown("""
    <div class="header-box">
        <h1 class="main-title">📊 Dashboard Analisis & Klasifikasi Stunting</h1>
        <p class="sub-title">Aplikasi ini menampilkan proses Preprocessing Data hingga Tahap Persiapan Modeling.</p>
    </div>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('pages/data_stunting.csv')
    return df

df_raw = load_data()

# 2. Tabs untuk memisahkan Analisis dan Prediksi
tab1, tab2 = st.tabs(["🛠️ Preparation", "📝 Modeling Klasifikasi"])

with tab1:
    st.header("📊 Exploratory Data Analysis (EDA) Data Stunting")
    # =====================================================
    # 0. Exploratory Data Analysis (EDA)
    # =====================================================
    st.caption("Tahap eksplorasi data sebelum preprocessing dan modeling")

    df = df_raw.copy()

    # =====================================================
    # PERBAIKAN TIPE DATA 
    # =====================================================

    num_cols_fix = ['Birth Weight', 'Birth Height', 'Weight', 'Height']
    cat_cols_fix = ['BB/U', 'TB/U', 'BB/TB']

    # konversi tanggal
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce')

    # konversi numerik & kategori
    for col in df.columns:
        if col in num_cols_fix:
            df[col] = df[col].astype('float')
        if col in cat_cols_fix:
            df[col] = df[col].astype('category')

    # =====================================================
    # PREVIEW DATA
    # =====================================================
    st.subheader("1.  Preview Dataset")
    st.dataframe(df.head(), use_container_width=True)

    # =====================================================
    # STATISTIK DESKRIPTIF
    # =====================================================
    st.subheader("2.  Statistik Deskriptif")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Statistik Umum (Mean, Std, Min, Max)**")
        st.dataframe(df.describe(), use_container_width=True)

    with col2:
        st.write("**Median Setiap Variabel Numerik**")
        median_df = df.median(numeric_only=True).reset_index()
        median_df.columns = ["Variabel", "Median"]
        st.table(median_df)

    # =====================================================
    # HISTOGRAM
    # =====================================================
    st.subheader("3.  Histogram Variabel Numerik")

    num_cols = df.select_dtypes(include="number").columns
    cols = st.columns(4)

    for i, col in enumerate(num_cols):
        with cols[i % 4]:
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=20, edgecolor="black")
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)

    # =====================================================
    # BOXPLOT (OUTLIER)
    # =====================================================
    st.subheader("4.  Deteksi Outlier (Boxplot)")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df[num_cols], ax=ax)
    ax.set_title("Boxplot Seluruh Variabel Numerik")
    st.pyplot(fig)

    st.caption(
        "Titik di luar whisker menunjukkan outlier yang berpotensi memengaruhi performa model."
    )

    # =====================================================
    # HEATMAP KORELASI
    # =====================================================
    st.subheader("5.  Heatmap Korelasi")

    fig, ax = plt.subplots(figsize=(6, 4))
    corr = df[num_cols].corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Korelasi Antar Variabel Numerik")
    st.pyplot(fig)

    # =====================================================
    # DISTRIBUSI LABEL TB/U
    # =====================================================
    st.subheader("6.  Distribusi Status Stunting (TB/U)")

    if "TB/U" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(
            data=df,
            x="TB/U",
            order=df["TB/U"].value_counts().index,
            ax=ax
        )
        ax.set_title("Distribusi Kategori TB/U")
        ax.set_xlabel("Kategori TB/U")
        ax.set_ylabel("Jumlah Anak")
        st.pyplot(fig)
    else:
        st.warning("Kolom TB/U tidak ditemukan dalam dataset.")

    # =====================================================
    # KESIMPULAN EDA
    # =====================================================
    st.subheader("📝 Kesimpulan EDA")

    st.markdown("""
    - Data memiliki variasi nilai pada **berat dan tinggi badan anak**.
    - Ditemukan **outlier** pada beberapa variabel numerik.
    - Terdapat hubungan (korelasi) antar variabel pertumbuhan anak.
    - Distribusi kelas **TB/U tidak seimbang**, sehingga perlu perhatian pada tahap modeling.
    - Hasil EDA ini menjadi dasar untuk **normalisasi, seleksi fitur, dan klasifikasi stunting**.
    """)

    st.divider()

    st.header("📊 Feature Engineering")
     # 7. Menampilkan Data Awal
    st.subheader("7. Data Awal")
    st.dataframe(df_raw.head())

    # 8. Pembersihan (Drop Kolom & Hitung Umur)
    # df = df_raw.copy()
    if 'No' in df.columns:
        df = df.drop('No', axis=1)
    
    # Hitung Umur
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce')
    measurement_date = pd.to_datetime('2024-02-01')
    df['Age_Years'] = (measurement_date - df['Date of Birth']).dt.days / 365.25
    
    # --- PROSES IMPUTASI MEAN UNTUK UMUR ---
    missing_count = df['Age_Years'].isnull().sum()
    mean_age = df['Age_Years'].mean()
    df['Age_Years'] = df['Age_Years'].fillna(mean_age)
    
    df = df.drop(['Date of Birth'], axis=1)

    st.subheader("8. Hasil Perhitungan Umur & Drop Kolom 'No'")
    
    # Pesan Informasi Imputasi
    st.info(f"💡 **Informasi Imputasi:** Nilai kosong pada fitur Age_Years diisi dengan Mean: **{mean_age:.2f} Tahun**.")

    st.write(f"Data sekarang memiliki kolom: `{', '.join(df.columns)}`")
    st.dataframe(df.head())

    # 9. Encoding Kategorikal
    st.subheader("9. Label Encoding (Ordinal)")
    
    df['BB/U_Encoded'] = df['BB/U'].astype(str).str.lower().str.strip().map(
        {'very less': 0, 'less': 1, 'normal body weight': 2, 'risk more': 3}
    ).fillna(2) 
    
    df['TB/U_Encoded'] = df['TB/U'].astype(str).str.lower().str.strip().map(
        {'very short': 0, 'short': 1}
    ).fillna(0)
    
    df['BB/TB_Encoded'] = df['BB/TB'].astype(str).str.lower().str.strip().map(
        {'undernutrition': 0, 'good nutrition': 1, 'risk of overnutrition': 2, 'overnutrition': 3}
    ).fillna(1)

    st.write("Data setelah Encoding (Siap untuk Modeling):")
    st.dataframe(df[['BB/U', 'BB/U_Encoded', 'TB/U', 'TB/U_Encoded', 'BB/TB', 'BB/TB_Encoded']].head())

    # 10. Matriks Korelasi
    st.subheader("10. Matriks Korelasi")
    
    col_kiri, col_tengah, col_kanan = st.columns([1, 2, 1])
    
    with col_tengah:
        fig, ax = plt.subplots(figsize=(6, 4))
        # Tetap menggunakan cmap='coolwarm' sesuai permintaan
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax, 
                    annot_kws={"size": 8}) 
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        st.pyplot(fig, use_container_width=False)

    st.divider()
    
    st.header("📊 Normalisasi dan Standarisasi")
    # 11. Skewness Distribusi Data Numerik
    st.subheader("11. Skewness Distribusi Data")

    num_cols = ['Birth Weight', 'Birth Height', 'Weight', 'Height', 'Age_Years']

    skew_values = df[num_cols].skew()

    # Tampilkan dalam bentuk tabel
    st.write("Nilai Skewness untuk setiap fitur numerik:")
    st.dataframe(skew_values.reset_index().rename(
            columns={'index': 'Fitur', 0: 'Skewness'}
        )
    )

    # 12. Pemeriksaan Tipe Data
    st.subheader("12. Pemeriksaan Tipe Data")

    dtypes_df = df.dtypes.reset_index()
    dtypes_df.columns = ["Kolom", "Tipe Data"]

    st.dataframe(dtypes_df, use_container_width=True)

    # 13. Pemeriksaan Nilai ≤ 0
    st.subheader("13. Pemeriksaan Nilai ≤ 0")

    cols = ['Birth Height','Birth Weight','Height','Weight', 'Age_Years']

    df_invalid = df[
        (df['Birth Height'] <= 0) |
        (df['Birth Weight'] <= 0) |
        (df['Height'] <= 0) |
        (df['Weight'] <= 0) |
        (df['Age_Years'] <= 0)
    ]

    if df_invalid.empty:
        st.success("✅ Semua nilai > 0, siap untuk Box-Cox.")
    else:
        st.warning(f"⚠️ Ditemukan {len(df_invalid)} baris ≤ 0.")
        st.dataframe(df_invalid[cols], use_container_width=True)

    # 14. Penanganan Nilai ≤ 0 untuk Birth Height & Birth Weight
    st.subheader("14. Penanganan Nilai ≤ 0 pada Birth Height & Birth Weight")

    # Ganti nilai 0 dengan mean kolom Birth Height
    mean_birth_height = df['Birth Height'].mean()
    df['Birth Height'] = df['Birth Height'].replace(0, mean_birth_height)

    # Ganti nilai 0 dengan mean kolom Birth Height & Birth Weight
    mean_values = df[['Birth Height', 'Birth Weight']].mean()
    df[['Birth Height', 'Birth Weight']] = df[['Birth Height', 'Birth Weight']].replace(0, mean_values)

    # Tampilkan hasil
    st.write("Data setelah penggantian nilai ≤ 0:")
    st.dataframe(df[['Birth Height','Birth Weight']].head(), use_container_width=True)

    # 15. Pemeriksaan Missing Value
    st.subheader("15. Pemeriksaan Missing Value")

    missing = df.isna().sum()
    missing = missing[missing > 0]  # hanya kolom yang ada missing

    if missing.empty:
        st.success("✅ Tidak ada missing value pada data.")
    else:
        st.warning("⚠️ Ditemukan missing value pada beberapa kolom:")
        st.dataframe(
            missing.reset_index().rename(columns={'index': 'Fitur', 0: 'Jumlah Missing'}),
            use_container_width=True
        )

    # 16. Imputasi Missing Value dengan Mean
    st.subheader("16. Imputasi Missing Value dengan Mean")

    df['Birth Height'].fillna(df['Birth Height'].mean(), inplace=True)
    df['Birth Weight'].fillna(df['Birth Weight'].mean(), inplace=True)

    st.write("Data setelah imputasi missing value:")
    st.dataframe(df[['Birth Height', 'Birth Weight']].head(), use_container_width=True)

    # 17. Validasi Missing Value
    st.subheader("17. Validasi Missing Value Setelah Imputasi")

    missing = df.isna().sum()
    # Tampilkan semua kolom beserta jumlah missing-nya (mirip Colab)
    missing_df = missing.reset_index()
    missing_df.columns = ["Fitur", "Jumlah Missing"]

    st.dataframe(missing_df, use_container_width=True)

    # 18. Transformasi Box-Cox
    from scipy.stats import boxcox

    st.subheader("18. Transformasi Box-Cox Variabel")

    num_cols_boxcox = ['Birth Weight', 'Birth Height', 'Weight', 'Height', 'Age_Years']
    lambda_dict = {}

    for col in num_cols_boxcox:
        # Box-Cox hanya bisa untuk nilai > 0, pastikan sudah bersih
        transformed_data, lambda_val = boxcox(df[col])
        df[col] = transformed_data
        lambda_dict[col] = lambda_val
        st.write(f"✅ Lambda optimal untuk **{col}**: {lambda_val:.4f}")

    # Tampilkan preview data setelah Box-Cox
    st.write("Data setelah Transformasi Box-Cox:")
    st.dataframe(df[num_cols_boxcox].head(), use_container_width=True)

    # 19. Pemeriksaan Outlier setelah Box-Cox
    st.subheader("19. Boxplot Setelah Box-Cox")

    # Boxplot setelah Box-Cox
    fig, ax = plt.subplots(figsize=(8, 5))
    df[['Weight', 'Height', 'Birth Weight', 'Birth Height', 'Age_Years']].boxplot(ax=ax)
    ax.set_title('Boxplot Weight & Height Setelah Box-Cox')
    st.pyplot(fig)

    # Histogram setelah Box-Cox
    st.subheader("📊 Histogram Setelah Box-Cox")
    fig, ax = plt.subplots(figsize=(8, 5))
    df[['Weight', 'Height', 'Birth Weight', 'Birth Height', 'Age_Years']].hist(ax=ax)
    ax.set_title('Boxplot Weight & Height Setelah Box-Cox')
    st.pyplot(fig)


    # Histogram setelah Box-Cox
    st.subheader("Data Normalization")

    normalization = MinMaxScaler()
    df[['Weight', 'Height', 'Birth Weight', 'Birth Height', 'Age_Years']] = normalization.fit_transform(df[['Weight', 'Height', 'Birth Weight', 'Birth Height', 'Age_Years']])
    st.dataframe(df[['Weight', 'Height', 'Birth Weight', 'Birth Height', 'Age_Years', 'TB/U']])

    # st.session_state.df = df[['Weight', 'Height', 'Birth Weight', 'Birth Height', 'Age_Years', 'TB/U']]

with tab2:
    st.header("Modeling Klasifikasi")
    st.write("Bagian ini masih dalam tahap pengembangan atau sengaja dikosongkan.")
    st.info("Silakan cek tab Preparation untuk melihat proses pengolahan data.")