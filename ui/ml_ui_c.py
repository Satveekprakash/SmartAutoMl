import streamlit as st
import pandas as pd

def show_classification_result(best_test_acc, best_model, results):

    # =========================
    # 1. QUALITY LABEL
    # =========================
    if best_test_acc >= 0.9:
        quality = "Excellent"
    elif best_test_acc >= 0.75:
        quality = "Good"
    elif best_test_acc >= 0.6:
        quality = "Average"
    else:
        quality = "Poor"

    # =========================
    # 2. TITLE
    # =========================
    st.markdown("### Classification Model Results")

    # =========================
    # 3. METRICS
    # =========================
    col1, col2 = st.columns(2)

    col1.metric("Best Test Accuracy", f"{best_test_acc:.4f}")
    col2.metric("Best Model", best_model)

    st.write("Quality:", quality)

    # =========================
    # 4. TABLE VIEW
    # =========================
    table_data = []

    for model_name, scores in results.items():
        table_data.append({
            "Model": model_name,
            "Train Accuracy": scores["train"],
            "Test Accuracy": scores["test"]
        })

    df = pd.DataFrame(table_data)

    st.markdown("#### Model Comparison")
    st.dataframe(df)

    # =========================
    # 5. INTERPRETATION
    # =========================
    with st.expander("How to interpret results"):

        st.write("""
        Train Accuracy:
        - Performance on training data

        Test Accuracy:
        - Performance on unseen data (most important)

        Overfitting:
        - High train accuracy but low test accuracy

        Good Model:
        - High test accuracy
        - Small gap between train and test
        """)