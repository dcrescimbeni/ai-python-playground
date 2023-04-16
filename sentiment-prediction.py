import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report
  
# Load tweets dataset
def start_model():
  print("Loading data...")
  data = pd.read_csv('tweet-sentiment.csv', names=["sentiment", "id", "date", "flag", "user", "text"])

  # Split the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(data["text"], data["sentiment"], test_size=0.2, random_state=42)

  # Vectorize the text data using the Bag of Words model
  print("Vectorizing text...")
  vectorizer = CountVectorizer()
  X_train_counts = vectorizer.fit_transform(X_train)
  X_test_counts = vectorizer.transform(X_test)

  # Train the Multinomial Naive Bayes classifier
  print("Training model...")
  clf = MultinomialNB()
  clf.fit(X_train_counts, y_train)
  print("Model ready!")

  return vectorizer, clf


# Model evaluation stats
# y_pred = clf.predict(X_test_counts)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# start model
vectorizer, clf = start_model()

# Get user input to analyze sentiment
while True:
  user_input = input("Enter a tweet: ")
  text_counts = vectorizer.transform([user_input])
  sentiment_prediction = clf.predict(text_counts)
  
  sentiment_string = ''
  if sentiment_prediction[0] == 0:
    sentiment_string = "negative"
  elif sentiment_prediction[0] == 4:
    sentiment_string = "positive"

  print("Sentiment: ", sentiment_string)