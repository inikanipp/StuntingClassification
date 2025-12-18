import streamlit as st
import pandas as pd
import streamlit.components.v1 as components


with open("styles/style.css") as f:
    css = f"<style>{f.read()}</style>"

st.title('Business Understanding')

con1 = st.container()
con2 = st.container()
con3 = st.container()
con4 = st.container()


with con1 :
    col1, col2 = st.columns([1,2])
    with col1 :
        st.subheader("Pengertian Stunting")
        st.write("Stunting adalah kondisi gagal tumbuh pada anak akibat kekurangan gizi kronis "
        "(malnutrisi) dan infeksi berulang, terutama dalam 1000 hari pertama kehidupan, yang ditandai dengan tinggi atau " \
        "panjang badan anak lebih pendek dari standar usianya, serta menghambat perkembangan otak," \
        "kognitif, dan meningkatkan risiko penyakit di kemudian hari. Ini bukan sekadar masalah fisik, tapi juga " \
        "gangguan perkembangan jangka panjang. ")
    with col2 :
        st.image("https://kesmas-id.com/wp-content/uploads/2024/07/Penting-Kenali-Stunting-Sebelum-Terlambat.jpg", width='stretch')

with con2 :
    col3, col4 = st.columns([1,1])
    with col3 :
        st.subheader("Latar Belakang")
        st.write("Stunting merupakan salah satu permasalahan gizi kronis yang masih menjadi fokus utama kesehatan masyarakat di Indonesia," \
        "khususnya di wilayah Kabupaten Lombok Timur, Nusa Tenggara Barat. Anak dengan status pendek dan sangat pendek berisiko mengalami" \
        "hambatan pertumbuhan fisik, perkembangan kognitif, serta produktivitas jangka panjang. ")
    with col4 :
        st.subheader("Permasalahan Bisnis")
        st.write("""
        Tenaga kesehatan dan pemangku kebijakan masih menghadapi tantangan dalam:

        1. Mengidentifikasi anak dengan status stunting berat (sangat pendek) secara dini
        2. Melakukan pemetaan risiko stunting berdasarkan data pengukuran
        3. Mengevaluasi efektivitas intervensi gizi yang telah dilakukan

        Tanpa dukungan sistem berbasis data, proses klasifikasi masih bersifat manual 
        dan berpotensi tidak konsisten.
        """)

with con3 :
    col3, col4 = st.columns([2,1])
    with col3 :
        st.subheader("Data Overview")
        st.dataframe(pd.read_csv('pages/data_stunting.csv'))
        st.write("""
        Dataset ini berasal dari pengukuran balita pada Februari 2024 (pra-intervensi) dan Agustus 2024 (pasca-intervensi) setelah dilakukan program pemberian makanan tambahan dan edukasi gizi.
        """)
    with col4 :
        st.subheader("Karakteristik Utama Dataset")
        st.write("""
        Sumber data: Pengukuran lapangan tenaga kesehatan

        1. Jumlah data: 490 catatan pengukuran
        2. Periode pengambilan: Februari & Agustus 2024
        3. Wilayah: Kabupaten Lombok Timur
        4. Tipe data: Data numerik, Data Kategorikal & label status stunting
        5. Tujuan penggunaan: Klasifikasi status stunting anak
                 
        """)

    with con4 :
        col5, col6 = st.columns([1,1])
        with col5 : 
            st.subheader("Manfaat")
            st.write("""
                Manfaat yang diharapkan dari sistem ini antara lain:

                ✅ Membantu deteksi dini stunting
                     
                ✅ Mendukung pengambilan keputusan berbasis data
                     
                ✅ Menjadi alat bantu evaluasi program intervensi gizi
                     
                ✅ Mempermudah visualisasi dan eksplorasi data stunting
                    
            """)
        with col6 : 
            st.subheader("Ruang Lingkup")
            st.write("""
                Ruang lingkup proyek ini dibatasi pada:

                1. Data balita di wilayah Lombok Timur
                2. Klasifikasi pendek vs sangat pendek
                3. Tidak mencakup diagnosis medis, hanya sebagai alat bantu analisis
                    
            """)


st.markdown(css, unsafe_allow_html=True)