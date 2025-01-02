import streamlit as st
import joblib
import pandas as pd
from utils import preprocess_text, tfidf, validate_input

import warnings
warnings.filterwarnings('ignore')

# Load model and data
model = joblib.load('model.pkl')
data1 = pd.read_csv('Symptom2Disease.csv')
data2 = pd.read_csv('Disease2Action.csv')
symptoms = data1['text']

# Set the app title
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.title('MedBuddy')

# Add a welcome message
st.write('Welcome to MedBuddy app! :tada::sparkles: This app is designed to predict disease based on symptoms and provide prescription / medical advice to user.	:hospital::stethoscope:')
st.divider()

# Create a text input
# user_input = st.text_input('May I know what are your symptoms?', 'Yellowing of skin and eyes, fatigue')
st.subheader('May I know what are your symptoms?')
user_input = st.chat_input('symptoms')

if user_input != None:
    st.write(user_input)

    # Text Preprocessing
    preprocessed_text = preprocess_text(user_input)
    preprocessed_symptoms = symptoms.apply(preprocess_text)

    is_valid, error_msg = validate_input(preprocessed_text, preprocessed_symptoms)
    if not is_valid:
        if error_msg == 'Not Valid':
            with st.container(border=True):
                st.markdown(':red[Input does not contain recognizable symptoms. Please provide valid symptoms.]')

        elif error_msg == 'Insufficient':
            with st.container(border=True):
                st.markdown(
                    ':red[Please provide more details about your condition so that I can better understand and assist you.]')
    else:
        # Transform the preprocessed symptom using the same vectorizer used during training
        symptom_tfidf = tfidf(preprocessed_text, preprocessed_symptoms)

        # Predict the disease
        predicted_disease = model.predict(symptom_tfidf)

        # Prescription / medical advice
        action = data2.loc[data2.disease == predicted_disease[0], 'action'].values[0]

        # Display the result
        with st.container(border=True):
            st.write('**Predicted Disease**	:ambulance::')
            st.markdown(predicted_disease[0])
            st.divider()
            st.write('**Prescription / Medical Advice**	:pill::')
            st.markdown(action)
