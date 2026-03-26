import streamlit as st
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Smart AutoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM STYLE ------------------
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #00D4FF;}
    .stButton>button {
        background: linear-gradient(90deg, #00D4FF, #007CF0);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("🤖 Smart AutoML System")
st.caption("Train Machine Learning models automatically with zero coding 🚀")

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Configuration")
file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])
target = st.sidebar.text_input("🎯 Target Column (write or it auto detect)")

model_type = st.sidebar.selectbox(
    "🧠 Select Model Type",
    ["MachineLearning"]
)

run_btn = st.sidebar.button("🚀 Run Model", key="run_btn")

# ------------------ STORE DATA ------------------
if file:
    data = pd.read_csv(file)
    st.session_state["data"] = data

# ------------------ MAIN ------------------
if "data" in st.session_state:

    data = st.session_state["data"]

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    st.info(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")

    # chat
    from chat import show_ai_assistant

    if "data" in st.session_state:
        show_ai_assistant(st.session_state["data"])
    # target predtiction or auto
    # target = target.strip()
    #
    # if target == "":
    #     try:
    #         from llm_utils import detect_target
    #
    #         target = detect_target(data)
    #         st.success(f"🎯 Auto detected target: {target}")
    #     except Exception as e:
    #         st.warning("⚠️ Auto target detection unavailable. Type the target.")
    #         target =target
    # ------------------ RUN MODEL ------------------
    if run_btn:
        st.session_state["run_clicked"] = True

    if st.session_state.get("run_clicked"):

        with st.spinner("⏳ Training model... Please wait"):
            model = None

            # -------- ML UNSUPERVISED --------
            if target == "NONE":
                st.success("🔍 Running ML Unsupervised")

                from unsupervised import run_unsupervised

                best_model_name, best_model, results, pipeline, cluster_summary = run_unsupervised(data)

                model = pipeline
                st.session_state["encoder"] = None

                from ui.ml_ui_u import show_clustering_results

                show_clustering_results(best_model_name, results, cluster_summary)
            # -------- ML SUPERVISED --------
            else:
                st.success("⚡ Running Machine Learning Supervised")

                from src.data_detector import detect_problem_type
                t = detect_problem_type(data, target)

                if t == "Regression":
                    from regression import run_regression_ml
                    best_model_name, best_score, results, pipeline = run_regression_ml(data, target)

                    model = pipeline
                    st.session_state["encoder"] = None

                    from ui.ml_ui_r import show_regression_result
                    show_regression_result(best_model_name, best_score, results)

                else:
                    from classification import run_classification_ml
                    acc, model_name, pipeline, results, target_encoder = run_classification_ml(data, target)

                    model = pipeline
                    st.session_state["encoder"] = target_encoder

                    from ui.ml_ui_c import show_classification_result
                    show_classification_result(acc, model_name, results)

            # ------------------ SAVE MODEL ------------------
            if model:
                st.session_state["model"] = model
                st.session_state["columns"] = data.drop(
                    columns=[target], errors="ignore"
                ).columns
                st.session_state["trained"] = True

                st.success("✅ Model trained successfully!")

# ------------------ PREDICTION ------------------
if st.session_state.get("trained"):

    st.divider()
    st.subheader("🔮 Make Prediction")

    input_data = {}
    cols = st.columns(3)

    for i, col in enumerate(st.session_state["columns"]):
        with cols[i % 3]:
            val = st.number_input(f"{col}",value=0.0)
            input_data[col] = val

    # ------------------ PREDICT BUTTON ------------------
    if st.button("🎯 Predict", key="predict_btn"):

        df = pd.DataFrame([input_data])

        model = st.session_state["model"]
        pred = model.predict(df)

        # ------------------ HANDLE OUTPUT ------------------
        encoder = st.session_state.get("encoder")

        if encoder is not None:
            pred = encoder.inverse_transform(pred)

        st.success(f"Prediction: {pred[0]}")
