#Example script to denmonstrate the process of using sklearn for training a classifier for sentiment analysis.
#This script also shows how to compact processes into a pipeline

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib


#The dataset is the IBDM reviews of movies. It is available for free on https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews 
# The data is converted into a json format: [ { "text": "....." "label": "positive" or "negative" }, ............. ]
# Load the data from the JSON file
with open('train_data.json', 'r') as file:
    data = pd.read_json(file)

# Use only 4400 examples (4000 for training and 400 for testing)
data = data.sample(4400, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=400, random_state=42)


#We use TF-IDF implementation in sklearn to vectorize the data check also:
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import HashingVectorize
# Convert text data into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(clf, 'random_forest_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Load the saved model and vectorizer
clf = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Take user input and classify
while True:
    user_input = input("\nEnter a sentence to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    user_input_vec = vectorizer.transform([user_input])
    prediction = clf.predict(user_input_vec)
    print(f"Predicted sentiment: {prediction[0]}")

#Let's try the same procedure but with using pipeline
# Create a pipeline with a TfidfVectorizer and RandomForestClassifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained pipeline
joblib.dump(pipeline, 'text_classification_pipeline.pkl')

# Load the saved pipeline
pipeline = joblib.load('text_classification_pipeline.pkl')

# Take user input and classify
while True:
    user_input = input("\nEnter a sentence to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    prediction = pipeline.predict([user_input])
    print(f"Predicted sentiment: {prediction[0]}")

