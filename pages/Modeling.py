import streamlit as st
with open("styles/style.css") as f:
    css = f"<style>{f.read()}</style>"

# =============================================================================================================

st.title("Modeling")




# =============================================================================================================
st.markdown(css, unsafe_allow_html=True)