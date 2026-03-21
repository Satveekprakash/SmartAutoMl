
import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
def show_ai_assistant(data):

    st.divider()
    st.subheader("🤖 AI Data Assistant")

    # -------- GROQ --------
    client = Groq(api_key=os.getenv("API_KEY"))

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
                model="llama-3.3-70b-versatile",
                messages=[  # type: ignore
                    {"role": "system", "content": TEMPLATE},
                    {"role": "user", "content": f"{data_text}\n\nQuestion: {q}"}
                ]
            )

        st.success("✅ Done")
        st.write(response.choices[0].message.content)