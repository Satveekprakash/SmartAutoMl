import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
def detect_target(data):

    client = Groq(api_key=os.getenv("API_KEY"))

    sample = data.sample(min(50, len(data)))
    data_text = sample.to_string()

    columns = list(data.columns)

    prompt = f"""
Dataset columns: {columns}

Task:
Choose BEST target column for prediction.
If no clear target → return NONE.

Rules:
- Only return column name OR NONE
- No explanation

Dataset preview:
{data_text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[  # type: ignore
            {"role": "user", "content": prompt}
        ]
    )

    result = response.choices[0].message.content.strip()

    # clean output
    if result.upper() == "NONE" or result not in columns:
        return None

    return result