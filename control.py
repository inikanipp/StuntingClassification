import streamlit as st

pages = {
    "BUSINESS UNDERSTANDING": [
        st.Page("pages/business_understanding.py", title="BUSINESS UNDERSTANDING")
    ],
    "DATA UNDERSTANDING": [
        st.Page("pages/data_under.py", title="DATA UNDERSTANDING")
    ],
    "DATA PREPARATION": [
        st.Page("pages/data_prep.py", title="DATA PREPARATION"),
        st.Page("pages/split.py", title="SPLIT DATA")
    ],
    "MODELING": [
        st.Page("pages/Modeling.py", title="MODELING")
    ],
    "EVALUATION": [
        st.Page("pages/evaluasi.py", title="EVALUASI")
    ],
    "PREDICTION": [
        st.Page("pages/prediction.py", title="PREDICTION")
    ],
}

pg = st.navigation(pages)
pg.run()
