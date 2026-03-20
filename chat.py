import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import os


def show_ai_assistant(data):

    st.divider()
    st.subheader("🤖 AI Data Assistant (Groq)")

    # ------------------ COLUMN INFO ------------------
    st.markdown("### 📋 Dataset Columns")

    for col in data.columns:
        st.write(f"- {col} ({data[col].dtype}, unique={data[col].nunique()})")

    # ------------------ GROQ LLM ------------------
    #GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_API_KEY ="REMOVED_KEY"
    llm = OpenAI(
        api_token=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",  # ✅ GROQ endpoint
        model="llama3-70b-8192"  # 🔥 best free model
    )

    # use small sample for speed
    df_ai = SmartDataframe(
        data.sample(min(50, len(data))),
        config={"llm": llm, "verbose": False}
    )

    # ------------------ TEMPLATE ------------------
    TEMPLATE = """
You are a data analyst.

Tasks:
1. Show important plots
2. Show relationships between features
3. Show strength (correlation or impact)
4. Identify important columns

Rules:
- ALWAYS show plots first
- Keep output SHORT
- Max 5 bullet points

Format:

Columns:
- important columns

Plots:
- generate plots

Insights:
- Feature A ↔ Feature B → Strong (0.85)

Importance:
- Most important feature:
- Strongest relationship:
- Least useful feature:

No long explanations.
"""

    # =========================
    # SMART ANALYSIS
    # =========================
    if st.button("🧠 Smart Analysis"):

        with st.spinner("Analyzing..."):
            response = df_ai.chat(
                TEMPLATE + "\nAnalyze dataset with best plots."
            )

        st.success("✅ Done")
        st.write(response)

    # =========================
    # CHAT
    # =========================
    st.subheader("💬 Ask your data")

    q = st.text_input("Ask (plot, correlation, feature importance)")

    if q:
        with st.spinner("Thinking..."):
            response = df_ai.chat(TEMPLATE + "\nQuestion: " + q)

        st.write(response)

    # =========================
    # QUICK ACTIONS
    # =========================
    st.subheader("⚡ Quick Actions")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("📈 Correlation"):
            st.write(df_ai.chat("Plot correlation heatmap and show strongest relations"))

    with c2:
        if st.button("🔗 Relationships"):
            st.write(df_ai.chat("Plot top relationships between features"))

    with c3:
        if st.button("🔥 Important Features"):
            st.write(df_ai.chat("Find most important features and their impact"))