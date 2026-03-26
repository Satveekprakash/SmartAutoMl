
import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
def show_ai_assistant(data):
 try:
    st.divider()
    st.subheader("🤖 AI Data Assistant")

    # -------- GROQ --------
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    st.write("Available secrets:", list(st.secrets.keys()))

    if not api_key:
        st.error("❌ GROQ_API_KEY not found")
        st.stop()

    client = Groq(api_key=api_key)

    # small sample (fast + efficient)
    sample = data.sample(min(50, len(data)))
    data_text = sample.to_string()

    # -------- TEMPLATE --------
    TEMPLATE = """
You are a data analyst.

Rules:
- Answer ONLY using given dataset
- Keep answers SHORT (3-5 points)
- Focus on patterns, relationships, insights
- No theory or outside knowledge

If unrelated → say:
"Only dataset-related questions allowed"
"""

    # ================= INPUT =================
    q = st.text_input("Ask about your dataset")

    # ================= RESPONSE =================
    if q:

        with st.spinner("Analyzing..."):

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", #llama-3.1-8b-instant,#llama-3.3-70b-versatile
                messages=[  # type: ignore
                    {"role": "system", "content": TEMPLATE},
                    {"role": "user", "content": f"{data_text}\n\nQuestion: {q}"}
                ]
            )

        st.success("✅ Done")
        st.write(response.choices[0].message.content)


 except Exception as e:

     st.error(f"AI Assistant Error:@Try next time{e}")