import streamlit as st

pages = {
    "Menu": [
        st.Page("pages/Modeling.py", title="Upload Data"),
        st.Page("pages/tes.py", title="tes"),
        st.Page("pages/Modeling_Split.py", title="Split"),
    ]
}

pg = st.navigation(pages)
pg.run()