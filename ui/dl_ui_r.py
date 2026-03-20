import streamlit as st

def show_regression_result(result):

    # =========================
    # 1. QUALITY BASED ON MAE
    # =========================
    mae = result.get("mae", None)

    if mae is None:
        quality = "Unknown"
    elif mae < 1:
        quality = "Excellent"
    elif mae < 5:
        quality = "Good"
    elif mae < 10:
        quality = "Average"
    else:
        quality = "Poor"

    # =========================
    # 2. TITLE
    # =========================
    st.markdown("### Regression Model Performance")

    # =========================
    # 3. METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{result['mae']:.4f}")
    col2.metric("MSE Loss", f"{result['mse_loss']:.4f}")
    col3.metric("Quality", quality)

    # =========================
    # 4. SUMMARY BOX
    # =========================
    st.info(f"""
    Best Configuration:
    - Neurons: {result['best_config']['neurons']}
    - Learning Rate: {result['best_config']['learning_rate']}

    Threshold Used: {result['threshold']}

    Features Used: {len(result['features'])}
    """)

    # =========================
    # 5. INTERPRETATION
    # =========================
    with st.expander("How to interpret results"):

        st.write("""
        MAE (Mean Absolute Error):
        - Average difference between prediction and actual value
        - Lower is better

        MSE Loss:
        - Training loss used during model optimization
        - Lower is better

        Quality Guide:
        - MAE < 1 → Excellent
        - MAE < 5 → Good
        - MAE < 10 → Average
        - MAE ≥ 10 → Poor
        """)