from flask import Flask, request, jsonify
import joblib
import re

def clean_text(tweet):
    tweet = re.sub(r"http\S+|www\S+", '', tweet)  # Remove URLs
    tweet = re.sub(r"@\w+|#", '', tweet)          # Remove mentions and hashtags
    tweet = re.sub(r"[^\w\s]", '', tweet)         # Remove punctuation
    tweet = tweet.lower().strip()
    return tweet

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['tweet']
    clean = clean_text(text)
    vect = vectorizer.transform([clean])
    pred = model.predict(vect)[0]
    return jsonify({'sentiment': pred})

if __name__ == '__main__':
    app.run()
