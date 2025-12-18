import streamlit as st
import pandas as pd
from components import knn_explanation_component 
from function.tuning import tuning
from function.modeling import modeling
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

with open("styles/style.css") as f:
    css = f"<style>{f.read()}</style>"

# =============================================================================================================

st.title("✅ Hyperparameter Tuning | KNN")



components.html("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <div id="carouselExampleSlidesOnly" class="carousel slide" data-bs-ride="carousel">
    <div class="carousel-inner" style="border-radius : 12px">
        <div class="carousel-item active">
        <img src="https://plus.unsplash.com/premium_photo-1681400641919-d5d03f6c0720?q=80&w=1121&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" class="d-block w-100" alt="...">
        </div>
        <div class="carousel-item">
        <img src="https://plus.unsplash.com/premium_photo-1722859331626-2214e69baa97?q=80&w=1169&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" class="d-block w-100" alt="...">
        </div>
        <div class="carousel-item">
        <img src="https://images.unsplash.com/photo-1637195141546-2469a5312504?q=80&w=1172&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" class="d-block w-100" alt="...">
        </div>
    </div>
    </div>
""",height=260)

if 'X_train' in st.session_state :
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    knn_explanation_component()


    hyperparams = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    con1 = st.container()
    con2 = st.container()

    with con1 :
        col1,col2 = st.columns([1,1])
        with col1 :
            st.success("""
                 **Parameter n_neighbors**:  
                *[1, 3, 5, 7, 9, 11, 13]*
            """)
        with col2 :
            st.success("""
                 **metrics**:  
                *['euclidean', 'manhattan']*
            """)
    with con1 :
        col3 = st.columns(1)[0]
        with col3 :
            st.success("""
                 **weights**:  
                *['uniform', 'distance']*
            """)

        if st.button("tuning", key='tuning') :
            best_params, best_score = tuning(X_train, y_train)
            con3 = st.container()
            con4 = st.container()

            with con3 :
                col4, col5 = st.columns([1,1])
                with col4 :
                    st.success(f"""
                        **Parameter metric**:  
                        *{best_params['metric']}*
                    """)
                with col5 :
                    st.success(f"""
                        **Parameter n_neighbors**:  
                        *{best_params['n_neighbors']}*
                    """)
            with con4 :
                col6, col7 = st.columns([1,1])
                with col6 :
                    st.success(f"""
                        **Parameter weights**:  
                        *{best_params['weights']}*
                    """)
                with col7 :
                    st.success(f"""
                        **Akurasi**:  
                        *{best_score:.4f}*
                    """)
            st.subheader("📈 Uji dengan Data Test")
           
            confussion, accuracy, class_report = modeling(X_train,y_train,X_test,y_test,best_params['n_neighbors'], best_params['metric'], best_params['weights'])
            con5 = st.container()

            st.session_state.confussion = confussion
            st.session_state.accuracy = accuracy
            st.session_state.class_report = class_report

            with con5 :
                col8, col9 = st.columns([5,1])
                with col8 :
                    classes = ['very short', 'short']
                    fig, ax = plt.subplots()
                    sns.heatmap(confussion,annot=True,xticklabels=classes, yticklabels=classes)
                    plt.tight_layout()
                    ax.set_ylabel('Prediksi')
                    ax.set_xlabel('Aktual')
                    
                    st.pyplot(fig)
                    
                with col9:
                    st.success(f"""
                        **akurasi**:  
                        *{accuracy:.4f}*
                    """)

            


                

else :
    st.error("Split Data Terlebih Dahulu !")


# =============================================================================================================
st.markdown(css, unsafe_allow_html=True)