import streamlit as st

def  show_classification_result(best_model, best_config, best_acc):

    # =========================
    # 1. QUALITY LABEL
    # =========================
    if best_acc >= 0.9:
        quality = "Excellent"
    elif best_acc >= 0.75:
        quality = "Good"
    elif best_acc >= 0.6:
        quality = "Average"
    else:
        quality = "Poor"

    # =========================
    # 2. TITLE
    # =========================
    st.markdown("### Model Result")

    # =========================
    # 3. METRICS
    # =========================
    col1, col2 = st.columns(2)

    col1.metric("Best Accuracy", f"{best_acc:.4f}")
    col2.metric("Quality", quality)

    # =========================
    # 4. CONFIG BOX
    # =========================
    st.info(f"""
    Best Configuration:
    - Neurons: {best_config[0]}
    - Learning Rate: {best_config[1]}
    """)

