import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from function.format_eval import format_eval


def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("File style.css tidak ditemukan. Menggunakan styling default.")

load_css("styles/style_lavel.css")


st.title("🔍 Evaluasi Model KNN")
st.markdown("""
Selamat datang di dashboard evaluasi performa model **K-Nearest Neighbor (KNN)**. 
Tahap ini mengukur akurasi model menggunakan data uji dengan metrik seperti Confusion Matrix, Classification Report, dan skor evaluasi lainnya.
""")

st.divider()

if 'accuracy' not in st.session_state :
    st.warning("Tidak Ada Hasil")

if 'accuracy' in st.session_state :

    st.markdown('<div class="overview-section">', unsafe_allow_html=True)
    st.header("📈 Overview")
    st.markdown("""
    - **Model**: K-Nearest Neighbor (KNN) dengan k=11.
    - **Dataset**: Data uji untuk klasifikasi biner (Kelas 0 dan Kelas 1).
    - **Tujuan**: Menganalisis performa model untuk memastikan keakuratan prediksi.
    """)
    st.image(
        "https://via.placeholder.com/800x200/ffe4e6/991b1b?text=KNN+Evaluation+Dashboard",
        width=800
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()


    st.markdown('<div class="confusion-section">', unsafe_allow_html=True)
    st.header("📊 Confusion Matrix")
    st.markdown("""
    Confusion Matrix membandingkan prediksi model dengan data aktual. Heatmap di bawah menunjukkan jumlah dan persentase untuk setiap kategori.
    """)


    cm = st.session_state.confussion


    cm_normalized = cm.astype('float') / cm.sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Prediksi 0", "Prediksi 1"], yticklabels=["Aktual 0", "Aktual 1"], ax=ax1)
    ax1.set_title("Confusion Matrix (Jumlah)")
    ax1.set_xlabel("Hasil Prediksi")
    ax1.set_ylabel("Data Aktual")


    sns.heatmap(cm_normalized, annot=True, fmt=".1f", cmap="Reds", xticklabels=["Prediksi 0", "Prediksi 1"], yticklabels=["Aktual 0", "Aktual 1"], ax=ax2)
    ax2.set_title("Confusion Matrix (Persentase)")
    ax2.set_xlabel("Hasil Prediksi")
    ax2.set_ylabel("Data Aktual")

    st.pyplot(fig)


    st.markdown("""
    **Penjelasan Grafik:**
    - **Heatmap Kiri (Jumlah)**: Menampilkan jumlah data aktual vs. prediksi. Diagonal utama (TN dan TP) menunjukkan prediksi benar, sedangkan diagonal lainnya (FP dan FN) menunjukkan kesalahan. Berguna untuk melihat distribusi kesalahan absolut.
    - **Heatmap Kanan (Persentase)**: Sama seperti kiri, tapi dalam persen dari total data. Membantu memahami proporsi kesalahan relatif, terutama jika dataset tidak seimbang.
    - **Mengapa Penting?**: Confusion Matrix membantu mengidentifikasi jenis kesalahan model (misalnya, lebih banyak FP atau FN), yang tidak terlihat dari akurasi saja.
    """)

    tn, fp, fn, tp = cm.ravel()
    st.info(f"""
    **Interpretasi:**
    - True Negative (TN): {tn} ({cm_normalized[0,0]:.1f}%)
    - False Positive (FP): {fp} ({cm_normalized[0,1]:.1f}%)
    - False Negative (FN): {fn} ({cm_normalized[1,0]:.1f}%)
    - True Positive (TP): {tp} ({cm_normalized[1,1]:.1f}%)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.header("📋 Classification Report")
    st.markdown("""
    Laporan ini menampilkan precision, recall, dan f1-score untuk setiap kelas. Visualisasi bar chart membantu perbandingan.
    """)

    report_data1 = st.session_state.class_report

    precision, recall, f1_score, support, index = format_eval(report_data1)

    report_data = {
        "precision": precision,
        "recall": recall,
        "f1-score": f1_score,
        "support": support
    }

    index = ["Kelas 0", "Kelas 1", "Accuracy", "Macro Avg", "Weighted Avg"]
    df_report = pd.DataFrame(report_data, index=index)
    st.dataframe(df_report)

    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = df_report.dropna().drop(columns=["support"])
    df_plot.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_title("Precision, Recall, dan F1-Score per Kelas")
    ax.set_ylabel("Skor")
    ax.set_xlabel("Kelas / Metrik")
    plt.xticks(rotation=45)
    st.pyplot(fig)


    st.markdown("""
    **Penjelasan Grafik:**
    - **Bar Chart**: Menampilkan precision (ketepatan prediksi positif), recall (kemampuan mendeteksi positif), dan f1-score (rata-rata harmonik precision dan recall) untuk setiap kelas dan rata-rata.
    - **Mengapa Penting?**: Membantu melihat performa model per kelas. Misalnya, jika recall Kelas 0 rendah, model sering melewatkan kelas tersebut. F1-score memberikan keseimbangan antara precision dan recall.
    - **Interpretasi**: Tinggi di semua metrik menunjukkan model baik, tapi perhatikan ketidakseimbangan antar kelas (misalnya, Kelas 1 lebih baik dari Kelas 0).
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()


    st.markdown('<div class="metrics-section">', unsafe_allow_html=True)
    st.header("📏 Metrik Evaluasi Model")
    st.markdown("Metrik utama model KNN berdasarkan data uji. Hover pada kartu untuk detail.")

    accuracy = 0.7576
    precision = 0.8182
    recall = 0.7576
    f1 = 0.6779

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}", help="Proporsi prediksi yang benar.")
    with col2:
        st.metric("Precision", f"{precision:.2%}", help="Ketepatan prediksi positif.")
    with col3:
        st.metric("Recall", f"{recall:.2%}", help="Kemampuan mendeteksi positif.")
    with col4:
        st.metric("F1-Score", f"{f1:.2%}", help="Rata-rata harmonik precision dan recall.")


    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [accuracy, precision, recall, f1]
    ax.bar(metrics, values, color=["#dc2626", "#b91c1c", "#991b1b", "#7f1d1d"])
    ax.set_title("Perbandingan Metrik Evaluasi")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2%}", ha="center")
    st.pyplot(fig)

    st.markdown("""
    **Penjelasan Grafik:**
    - **Bar Chart**: Membandingkan empat metrik utama (Accuracy, Precision, Recall, F1-Score) dalam satu visualisasi. Tinggi bar menunjukkan performa yang lebih baik.
    - **Mengapa Penting?**: Memberikan gambaran keseluruhan performa model. Accuracy tinggi tapi precision rendah bisa berarti model menebak positif terlalu sering.
    - **Interpretasi**: Jika semua bar tinggi (di atas 70%), model cukup baik. F1-Score yang rendah menunjukkan ketidakseimbangan antara precision dan recall.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()


    st.markdown('<div class="interpretation-section">', unsafe_allow_html=True)
    st.header("🎯 Interpretasi Hasil Evaluasi")
    with st.expander("Klik untuk melihat interpretasi lengkap"):
        st.success("""
        ### 📊 Analisis Performa Model
        - **Accuracy (75.76%)**: Model berhasil mengklasifikasikan sebagian besar data uji dengan benar, menunjukkan performa yang solid.
        - **Precision (81.82%)**: Tinggi, berarti model jarang salah memprediksi kelas positif.
        - **Recall (75.76%)**: Baik, namun ada ruang untuk perbaikan dalam mendeteksi semua kasus positif.
        - **F1-Score (67.79%)**: Keseimbangan yang wajar antara precision dan recall.

        📌 **Kesimpulan**:  
        Model KNN (k=13) **layak digunakan** untuk klasifikasi ini, meskipun bisa dioptimalkan lebih lanjut (misalnya, tuning hyperparameter atau preprocessing data).
        """)
    st.markdown('</div>', unsafe_allow_html=True)


    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #7f1d1d;">
        <p>Dashboard dibuat dengan menggunakan Streamlit. Data berdasarkan hasil evaluasi Colab.</p>
    </div>
    """, unsafe_allow_html=True)