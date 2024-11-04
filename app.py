import pickle
import numpy as np
import streamlit as st

# Load the model and scaler
with open('classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Display app header
st.markdown("""
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Banking Marketing Prediction</h2>
    </div>
""", unsafe_allow_html=True)

st.subheader("Predict if a client will subscribe to a term deposit")

# Explanation of the app and the data
st.markdown("""
    ## About the app

    This app predicts if a client will subscribe to a term deposit.

    ## About the data

    The data is related to direct marketing campaigns of a Portuguese banking institution, where the campaigns were based on phone calls. The goal is to predict whether a client will subscribe to a bank term deposit based on specific features.

    ## About the model

    The model was trained using a Random Forest Classifier.
""")

st.text("Please fill the form below")

# User input fields with limits
age = st.slider('Age', 16, 100, 16)
day = st.number_input("Day of the month [1-31]", min_value=1, max_value=31)
balance = st.number_input("Yearly balance in euros [-10k to 1M]", min_value=-10000, max_value=1000000)
campaign = st.number_input("Contacts performed during this campaign for this client [0-10]", min_value=0, max_value=10)
duration = st.number_input("Duration of last contact in seconds", min_value=0, max_value=5000)

# Ask if the client has been contacted in a previous campaign
previous_contact = st.selectbox("Has the client been contacted in a previous campaign?", ("Yes", "No"))

# If the client has been contacted before, prompt for 'pdays' and 'poutcome_success'
if previous_contact == "Yes":
    pdays = st.number_input("Days since last contact from a previous campaign", min_value=0, max_value=31)
    poutcome_success = st.selectbox("Was the previous campaign a success?", ("Yes", "No"))
    previous = 1  # Client has previous contact
else:
    pdays = -1
    poutcome_success = "No"
    previous = 0  # No previous contact

# Prepare feature values for scaling
scaling_features = {
    'age': age,
    'balance': balance,
    'day': day,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous
}

# Order the features according to scaler's expected order
scaling_order = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
ordered_scaling_features = np.array([[scaling_features[feature] for feature in scaling_order]])

# Prediction function
def make_prediction(ordered_scaling_features, poutcome_success):
    # Scale the features
    scaled_features = scaler.transform(ordered_scaling_features)
    
    # Reorder and select features according to the model's expected input order
    model_order = ['pdays', 'day', 'poutcome_success', 'age', 'balance', 'duration']
    model_features = np.zeros((1, len(model_order)))
    
    for i, feature in enumerate(model_order):
        if feature == 'poutcome_success':
            model_features[0, i] = 1 if poutcome_success == "Yes" else 0
        else:
            index = scaling_order.index(feature)
            model_features[0, i] = scaled_features[0, index]

    # Make predictions
    prediction = classifier.predict(model_features)[0]
    probabilities = classifier.predict_proba(model_features)[0]
    
    return prediction, probabilities

# Display prediction result when button is clicked
if st.button("Predict"): 
    result, probabilities = make_prediction(ordered_scaling_features, poutcome_success)
    st.success(f'The client will {"accept" if result == "yes" else "reject"} the offer')
    st.write(f"Probability of acceptance: {probabilities[1]:.2f}")
    st.write(f"Probability of rejection: {probabilities[0]:.2f}")










