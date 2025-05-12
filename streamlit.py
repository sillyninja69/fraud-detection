import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

def preprocess_data(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    # Log transform the Amount column to reduce skewness
    X['Amount'] = np.log1p(X['Amount'])
    return X, y

@st.cache(allow_output_mutation=True)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

def main():
    st.title("Credit Card Fraud Detection")
    st.write("""
    This is a simple credit card fraud detection app using Logistic Regression.
    Enter transaction details below to check if it is likely fraudulent.
    """)

    data = load_data()
    X, y = preprocess_data(data)
    model, X_test, y_test, y_pred = train_model(X, y)

    st.subheader("Model Performance on Test Data")
    st.text("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    st.text("Classification Report:")
    report = classification_report(y_test, y_pred)
    st.text(report)

    st.subheader("Make a Prediction")

    # Input fields for user
    amount = st.number_input("Transaction Amount (USD)", min_value=0.0, value=1.0, step=0.01)
    v1 = st.number_input("V1", value=0.0, step=0.01)
    v2 = st.number_input("V2", value=0.0, step=0.01)
    v3 = st.number_input("V3", value=0.0, step=0.01)
    v4 = st.number_input("V4", value=0.0, step=0.01)
    v5 = st.number_input("V5", value=0.0, step=0.01)

    # Other features V6 to V28 set to 0 for simplicity
    # Create input vector in correct order
    features_order = [f'V{i}' for i in range(1,29)] + ['Amount']

    input_features = {}
    for i in range(1,6):
        input_features[f'V{i}'] = locals()[f'v{i}']
    for i in range(6,29):
        input_features[f'V{i}'] = 0.0
    # Log transform amount same as training
    input_features['Amount'] = np.log1p(amount)

    input_vector = np.array([input_features[f] for f in features_order]).reshape(1, -1)

    if st.button("Predict"):
        prediction = model.predict(input_vector)[0]
        pred_prob = model.predict_proba(input_vector)[0][1]
        if prediction == 1:
            st.error(f"Warning! The transaction is predicted as FRAUDULENT with probability {pred_prob:.2f}")
        else:
            st.success(f"The transaction is predicted as LEGITIMATE with fraud probability {pred_prob:.2f}")

if __name__ == "__main__":
    main()

