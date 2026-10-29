import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="E-Learning Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- FILE PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"

MODEL_PATH = MODEL_DIR / "extra_trees_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
INFO_PATH = MODEL_DIR / "model_info.json"


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    """Load the trained model, preprocessor and model information."""

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    info = {}
    if INFO_PATH.exists():
        with INFO_PATH.open("r", encoding="utf-8") as file:
            info = json.load(file)

    return model, preprocessor, info


# ---------------- APP HEADER ----------------
st.title("🎓 E-Learning Dropout Prediction")

st.write(
    "A machine learning application for identifying "
    "students who may be at risk of dropping out."
)

# ---------------- CHECK MODEL FILES ----------------
missing_files = [
    path.name
    for path in [MODEL_PATH, PREPROCESSOR_PATH]
    if not path.is_file()
]

if missing_files:
    st.error(
        "Required model files were not found in the "
        "'saved_model' folder."
    )

    st.write("Missing files:")
    for filename in missing_files:
        st.write(f"- {filename}")

    st.info(
        "Ensure that the files are uploaded to GitHub "
        "inside the saved_model folder and that Git LFS "
        "has uploaded the actual model files."
    )
    st.stop()

# ---------------- INITIALIZE MODEL ----------------
try:
    model, preprocessor, info = load_artifacts()

except Exception as e:
    st.error("The model could not be loaded.")
    st.code(str(e))
    st.info(
        "Check the Streamlit logs and verify that the "
        "saved model files are valid and compatible "
        "with the installed Python libraries."
    )
    st.stop()


# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Dashboard",
        "Predict Dropout Risk",
        "About Project"
    ]
)


# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.subheader("Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset Size",
        info.get("dataset_rows", "N/A")
    )

    accuracy = info.get("test_accuracy_percent", "N/A")
    accuracy_display = (
        f"{accuracy}%"
        if accuracy != "N/A" and accuracy is not None
        else "N/A"
    )

    col2.metric(
        "Reported Test Accuracy",
        accuracy_display
    )

    col3.metric(
        "Machine Learning Model",
        info.get("model_name", "Extra Trees")
    )

    st.info(
        "The dataset used for model development is a hybrid dataset "
        "combining real-world student data with added instances "
        "showing student dropout risk. "
        "These figures describe the project dataset and model evaluation."
    )

    st.subheader("Supported Risk Categories")

    a, b, c = st.columns(3)

    a.success("LOW RISK")
    b.warning("MEDIUM RISK")
    c.error("HIGH RISK")

    st.write(
        "Select 'Predict Dropout Risk' from the sidebar "
        "to evaluate an individual student profile."
    )


# ---------------- PREDICTION PAGE ----------------
elif page == "Predict Dropout Risk":

    st.subheader("Student Activity Details")

    st.write(
        "Enter the student's e-learning activity details "
        "to estimate the dropout-risk category."
    )

    with st.form("student_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            course_category = st.selectbox(
                "Course Category",
                ["Programming", "Data Science", "Other"]
            )

            login_frequency = st.number_input(
                "Logins per Week",
                min_value=1,
                max_value=50,
                value=3
            )

            avg_session_minutes = st.number_input(
                "Average Session Duration (minutes)",
                min_value=1,
                max_value=600,
                value=25
            )

            course_progress = st.slider(
                "Course Progress (%)",
                min_value=0,
                max_value=100,
                value=20
            )

            assignments_completed = st.number_input(
                "Assignments Completed",
                min_value=0,
                max_value=100,
                value=2
            )

        with col2:

            quiz_average = st.slider(
                "Average Quiz Score (%)",
                min_value=0,
                max_value=100,
                value=45
            )

            forum_interactions = st.number_input(
                "Forum Interactions",
                min_value=0,
                max_value=1000,
                value=5
            )

            video_watch = st.slider(
                "Video Watch Percentage (%)",
                min_value=0,
                max_value=100,
                value=25
            )

            device_type = st.selectbox(
                "Device Type",
                ["Mobile", "Desktop", "Tablet"]
            )

        submitted = st.form_submit_button(
            "Predict Dropout Risk",
            use_container_width=True
        )

    if submitted:

        avg_minutes_per_login = round(
            avg_session_minutes / login_frequency,
            4
        )

        # Feature names must match the training pipeline.
        student = pd.DataFrame([{
            "course_category": course_category,
            "login_frequency_per_week": login_frequency,
            "avg_session_minutes": avg_session_minutes,
            "course_progress_percent": course_progress,
            "assignments_completed": assignments_completed,
            "quiz_average_score": quiz_average,
            "forum_interactions": forum_interactions,
            "video_watch_percent": video_watch,
            "device_type": device_type,
            "avg_minutes_per_login": avg_minutes_per_login
        }])

        try:

            processed_student = preprocessor.transform(student)
            prediction = model.predict(processed_student)[0]

            risk = str(prediction).strip().lower()

            st.divider()
            st.subheader("Prediction Result")

            if risk == "high":
                st.error(f"Predicted Risk: {prediction}")

            elif risk == "medium":
                st.warning(f"Predicted Risk: {prediction}")

            elif risk == "low":
                st.success(f"Predicted Risk: {prediction}")

            else:
                st.info(f"Predicted Category: {prediction}")

            st.caption(
                "This is a model estimate, not a definitive "
                "judgment about a student's future."
            )

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    processed_student
                )[0]

                st.subheader("Model Probability Estimates")

                results = pd.DataFrame({
                    "Risk Category": model.classes_,
                    "Estimated Probability (%)": [
                        round(float(probability) * 100, 2)
                        for probability in probabilities
                    ]
                })

                st.bar_chart(
                    results.set_index("Risk Category")
                    ["Estimated Probability (%)"]
                )

                st.dataframe(
                    results,
                    hide_index=True,
                    use_container_width=True
                )

            st.subheader("Submitted Student Details")

            st.dataframe(
                student,
                hide_index=True,
                use_container_width=True
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")

            st.info(
                "Verify that the input feature names, categories "
                "and preprocessing steps match those used during "
                "model training."
            )


# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":

    st.subheader("About the Project")

    st.write("""
    **Project:** Predictive Analytics for E-Learning Dropouts

    **Objective:** Identify patterns in student engagement
    that may be associated with dropout risk.

    **Algorithm:** Extra Trees Classifier

    **Preprocessing:** The saved preprocessing pipeline is
    used to transform student activity data.

    **Target:** dropout_risk

    **Dataset:** Hybrid dataset.
    """)