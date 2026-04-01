import streamlit as st
import pickle

st.set_page_config(page_title="Spam Detector")

st.title("📩 SMS Spam Classifier")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

message = st.text_area("Enter your message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]

        if prediction == 1:
            st.error("Spam 🚨")
        else:
            st.success("Not Spam ✅")