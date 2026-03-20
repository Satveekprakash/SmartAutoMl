import streamlit as st
import pandas as pd

def show_regression_result(best_model, best_score, results):

    # =========================
    # 1. QUALITY LABEL
    # =========================
    if best_score >= 0.9:
        quality = "Excellent"
    elif best_score >= 0.75:
        quality = "Good"
    elif best_score >= 0.5:
        quality = "Average"
    else:
        quality = "Poor"

    # =========================
    # 2. TITLE
    # =========================
    st.markdown("### Regression Model Results")

    # =========================
    # 3. METRICS
    # =========================
    col1, col2 = st.columns(2)

    col1.metric("Best R2 Score", f"{best_score:.4f}")
    col2.metric("Best Model", best_model)

    st.write("Quality:", quality)

    # =========================
    # 4. TABLE VIEW (same style)
    # =========================
    table_data = []

    for model_name, scores in results.items():
        table_data.append({
            "Model": model_name,
            "Train R2": scores["train_r2"],
            "Test R2": scores["test_r2"],
            "RMSE": scores["rmse"]
        })

    df = pd.DataFrame(table_data)

    st.markdown("#### Model Comparison")
    st.dataframe(df)

    # =========================
    # 5. INTERPRETATION
    # =========================
    with st.expander("How to interpret results"):

        st.write("""
        Train R2:
        - Performance on training data

        Test R2:
        - Performance on unseen data (most important)

        RMSE:
        - Prediction error (lower is better)

        Overfitting:
        - High Train R2 but low Test R2

        Good Model:
        - High Test R2
        - Low RMSE
        - Small gap between train and test
        """)