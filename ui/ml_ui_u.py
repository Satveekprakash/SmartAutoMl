import streamlit as st


def show_clustering_results(best_model, results,cluster_summary=None):
    st.markdown("###  Clustering Model Comparison")

    # =========================
    # 1. METRICS (SIDE BY SIDE)
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("KMeans", f"{results['KMeans']:.4f}")
    col2.metric("DBSCAN", f"{results['DBSCAN']:.4f}")
    col3.metric("Agglomerative", f"{results['Agglomerative']:.4f}")

    # =========================
    # 2. BEST MODEL HIGHLIGHT
    # =========================
    st.success(f"Best Model: **{best_model}**")

    # =========================
    # 3. SIMPLE PERFORMANCE BOX
    # =========================
    st.info(f"""
    **Model Scores (Higher is Better):**

    - KMeans: {results['KMeans']:.4f}  
    - DBSCAN: {results['DBSCAN']:.4f}  
    - Agglomerative: {results['Agglomerative']:.4f}  

      Selected Best Model: **{best_model}**
    """)

    # =========================
    # 4. OPTIONAL INTERPRETATION
    # =========================
    with st.expander("How to interpret scores"):
        st.write("""
        - Higher score → Better clustering  
        - Scores usually come from Silhouette Score  
        - Range:
            - 0.7 → Excellent  
            - 0.5 → Good  
            - 0.25 → Weak  
            - <0.25 → Poor  
        """)
        # =========================
        # 5. CLUSTER UNDERSTANDING
        # =========================
    if cluster_summary is not None:
        st.markdown("### 📊 Cluster Characteristics")

        st.write("Each cluster represents a group of similar data points.")

        st.dataframe(cluster_summary)

        st.caption("Values represent average feature values for each cluster.")