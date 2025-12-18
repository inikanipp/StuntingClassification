import streamlit as st

def knn_explanation_component():
   
    st.info("""
    📌 **Pengertian**:  
    *Hyperparameter tuning adalah proses mencari kombinasi parameter terbaik yang tidak dipelajari
            langsung oleh model dari data, tetapi ditentukan sebelum proses training.*
    """)

    st.subheader("📈 Hyperparameter Tuning")
    st.info("""
    *Pada tahap ini dilakukan hyperparameter tuning untuk menentukan 
    parameter terbaik pada model K-Nearest Neighbors. K-Nearest Neighbors
    dipilih karena pada uji coba skenario dengan menggunakan beberapa 
    model klasifikasi, K-Nearest Neigbors mendapatkan hasil yang lebih 
    tinggi.*
    """)
    st.subheader("🔄 Skenario Hyperparameter Tuning")


    