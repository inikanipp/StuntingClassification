import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.title("Data Preparation – Stratified Train-Test Split")

file = st.file_uploader("Upload dataset (CSV)", type="csv")

if file:
    df = pd.read_csv(file)
    st.dataframe(df)

    target = st.selectbox("Pilih kolom target", df.columns)
    test_size = st.slider("Ukuran data test (%)", 10, 50, 20)

    if st.button("Split Data"):
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=42,
            stratify=y
        )

        st.success("Data berhasil di-split secara stratified")
        st.write("Jumlah data train (latih) :", X_train.shape[0])
        st.write("Jumlah data test (uji) :", X_test.shape[0])

        st.write("Distribusi kelas (y_train)")
        st.write(y_train.value_counts(normalize=True))

        st.write("Distribusi kelas - Data Uji (y_test)")
        st.write(y_test.value_counts(normalize=True))