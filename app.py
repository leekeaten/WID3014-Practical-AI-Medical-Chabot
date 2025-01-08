import warnings

import joblib
import pandas as pd
import streamlit as st

from utils import preprocess_text, tfidf, validate_input

warnings.filterwarnings("ignore")

# Load model and data with error handling
try:
    model = joblib.load("model.pkl")
except Exception as e:  # noqa: BLE001
    st.error("Error loading the model: " + str(e))
    st.stop()  # Stop the app if model loading fails

try:
    data1 = pd.read_csv("Symptom2Disease.csv")
    data2 = pd.read_csv("Disease2Action.csv")
except Exception as e:  # noqa: BLE001
    st.error("Error loading data files: " + str(e))
    st.stop()  # Stop the app if data loading fails

symptoms = data1["text"]

# Set the app title
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.title("MedBuddy")

# Add a welcome message
st.write(
    "Welcome to MedBuddy app! :tada::sparkles: This app is designed to predict disease based on symptoms and provide prescription / medical advice to user.	:hospital::stethoscope:",
)
st.divider()

# Create a text input
st.subheader("May I know what are your symptoms?")
user_input = st.chat_input("symptoms")

if user_input is not None:
    st.write(user_input)

    # Text Preprocessing
    try:
        preprocessed_text = preprocess_text(user_input)
    except Exception as e:  # noqa: BLE001
        st.error(f"Error preprocessing input: {e}")
        st.stop()  # Stop the app if preprocessing fails

    preprocessed_symptoms = symptoms.apply(preprocess_text)

    is_valid, error_msg = validate_input(preprocessed_text, preprocessed_symptoms)
    if not is_valid:
        if error_msg == "Not Valid":
            with st.container(border=True):
                st.markdown(
                    ":red[Input does not contain recognizable symptoms.\
                    Please provide valid symptoms.]",
                )

        elif error_msg == "Insufficient":
            with st.container(border=True):
                st.markdown(
                    ":red[Please provide more details about your condition so that\
                    I can better understand and assist you.]",
                )
    else:
        # Transform the preprocessed symptom using the same vectorizer during training
        try:
            symptom_tfidf = tfidf(preprocessed_text, preprocessed_symptoms)
        except Exception as e:  # noqa: BLE001
            st.error(f"Error transforming symptoms: {e}")
            st.stop()  # Stop the app if transformation fails

        # Predict the disease
        try:
            predicted_disease = model.predict(symptom_tfidf)
        except Exception as e:  # noqa: BLE001
            st.error(f"Error predicting disease: {e}")
            st.stop()  # Stop the app if prediction fails

        # Prescription / medical advice
        try:
            action = data2.loc[data2.disease == predicted_disease[0], "action"].values[0]  # noqa: E501, PD011
        except Exception as e:  # noqa: BLE001
            st.error(f"Error retrieving prescription/advice: {e}")
            st.stop()  # Stop the app if prescription retrieval fails

        # Display the result
        with st.container(border=True):
            st.write("**Predicted Disease**	:ambulance::")
            st.markdown(predicted_disease[0])
            st.divider()
            st.write("**Prescription / Medical Advice**	:pill::")
            st.markdown(action)
