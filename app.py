from flask import Flask, request, render_template
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Get message from form
    message = request.form['message']

    # Transform message (VERY IMPORTANT)
    transformed_message = vectorizer.transform([message])

    # Predict
    prediction = model.predict(transformed_message)[0]

    if prediction == 1:
        result = "Spam 🚨"
    else:
        result = "Not Spam ✅"

    return render_template(
        'index.html',
        prediction_text=f"Prediction: {result}"
    )


if __name__ == "__main__":
    app.run(debug=True)