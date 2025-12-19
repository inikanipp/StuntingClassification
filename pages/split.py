import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split


# STYLE
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #e8f5e9, #ffffff);
}

.block-container {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0px 6px 14px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    color: #1b5e20;
}

h3 {
    color: #2e7d32;
}

.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.title("📊 Data Preparation")
st.markdown(
    "<p style='text-align:center; color:#555;'>Stratified Train-Test Split pada Data Stunting</p>",
    unsafe_allow_html=True
)

st.divider()

# LOAD DATA
DATA_PATH = 'pages/after_pre.csv'

TARGET_COL = "TB/U"

try:
    df = pd.read_csv('pages/after_pre.csv').drop(columns=['Unnamed: 0'])
    st.success("Dataset berhasil dimuat")
    
    with st.expander("🔍 Lihat Dataset"):
        st.dataframe(df, use_container_width=True)

except FileNotFoundError:
    st.error(f"File dataset '{DATA_PATH}' tidak ditemukan!")
    st.stop()

# SET TARGET & SPLIT
if TARGET_COL not in df.columns:
    st.error(f"Kolom target '{TARGET_COL}' tidak ditemukan!")
    st.stop()

st.info(f"🎯 Target Klasifikasi: **{TARGET_COL}**")

test_size = st.slider(
    "📌 Persentase Data Uji",
    min_value=10,
    max_value=50,
    value=20
)

st.divider()

# SPLIT PROCESS
if st.button("🚀 Proses Split Data"):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=42,
        stratify=y
    )

    st.success("✅ Data berhasil di-split secara stratified")
    col1, col2 = st.columns(2)
    col1.metric("📂 Data Latih", X_train.shape[0])
    col2.metric("📁 Data Uji", X_test.shape[0])

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📊 Distribusi Kelas (y_Train)")
        st.dataframe(
            y_train.value_counts(normalize=True).reset_index(),
            use_container_width=True
        )

    with col4:
        st.subheader("📊 Distribusi Kelas (y_Test)")
        st.dataframe(
            y_test.value_counts(normalize=True).reset_index(),
            use_container_width=True
        )
    if 'X_train' not in st.session_state :
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
    if 'X_train' in st.session_state :
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
