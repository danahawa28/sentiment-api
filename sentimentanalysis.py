import re
import pandas as pd

df = pd.read_csv("training.1600000.processed.noemoticon.csv",
                 encoding="latin-1", header=None)

df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = df[['target', 'text']]

#Preprocessing
def clean_text(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#','', tweet)  # Remove mentions and hashtags
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuation
    tweet = tweet.lower().strip()
    return tweet

df['clean_text'] = df['text'].apply(clean_text)


#Convert sentiment labels
def map_label(val):
    return {0: 'negative', 2: 'neutral', 4: 'positive'}.get(val)

df['sentiment'] = df['target'].apply(map_label)

#Model training
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred, average='weighted'))

import joblib

# Save the model and vectorizer
joblib.dump(model, "logistic_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


