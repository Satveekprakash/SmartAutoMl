import streamlit as st

def show_unsupervised_result(result):

    # =========================
    # 1. CALCULATE ACCURACY
    # =========================
    accuracy = 100 * (1 - result["reconstruction_error"])
    accuracy = max(0, min(100, accuracy))

    # =========================
    # 2. QUALITY LABEL
    # =========================
    if accuracy >= 90:
        quality = "Excellent "
    elif accuracy >= 75:
        quality = "Good "
    elif accuracy >= 60:
        quality = "Average "
    else:
        quality = "Poor "

    # =========================
    # 3. TITLE
    # =========================
    st.markdown("### Model Performance")

    # =========================
    # 4. METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.2f}%")
    col2.metric("Error", f"{result['reconstruction_error']:.6f}")
    col3.metric("Quality", quality)

    # =========================
    # 5. SUMMARY
    # =========================
    st.info(f"""
    **Best Config:**  
    - Neurons: {result['best_config']['neurons']}  
    - Learning Rate: {result['best_config']['learning_rate']}  

    **Features Used:** {len(result['features_used'])}  

    **Training Loss:** {result['best_loss']:.6f}
    """)

    # =========================
    # 6.  INTERPRETATION GUIDE
    # =========================
    with st.expander("How to Interpret This Model"):

        st.write("### What do these metrics mean?")

        st.write("""
        **Accuracy (%)**
        - Represents how well the model reconstructs data  
        - Higher = better learning  

        **Reconstruction Error**
        - Difference between input and output  
        - Lower = better model  

        **Quality**
        - Overall performance label based on accuracy  
        """)

        st.write("### Accuracy Guide")

        st.write("""
        - **90–100 → Excellent**  
        - **75–90 → Good** 
        - **60–75 → Average**   
        - **<60 → Poor** 
        """)

        st.write("###  When is model good?")

        st.write("""
        ✔ High accuracy (>75%)  
        ✔ Low reconstruction error  
        ✔ Stable results across runs  

           If accuracy is low:
        - Data may need cleaning  
        - Features may not be useful  
        - Model needs tuning  
        """)