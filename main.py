import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

print("=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)

print("Shape:", df.shape)

# Remove unnecessary columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("\n" + "=" * 50)
print("CLEANED DATASET")
print("=" * 50)

print("\nShape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nClass Distribution:")
print(df['label'].value_counts())

df.to_csv("outputs/cleaned_spam.csv", index=False)

# Convert labels into numbers

df['label_num'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nLabel Encoding:")
print(df[['label', 'label_num']].head())

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join(char for char in text if char not in string.punctuation)

    # Tokenization
    words = text.split()

    # Remove stopwords and apply stemming
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df['processed_message'] = df['message'].apply(preprocess_text)
print("\nProcessed Messages:")
print(df[['message', 'processed_message']].head())

print("\nCreating TF-IDF Features...")

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

X = tfidf.fit_transform(df['processed_message'])

y = df['label_num']

print("Feature Matrix Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Naive Bayes Model...")

model = MultinomialNB()

model.fit(X_train, y_train)

print("Model Training Completed!")

y_pred = model.predict(X_test)

print("\nMODEL EVALUATION")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Ham', 'Spam'],
    yticklabels=['Ham', 'Spam']
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")

plt.show()

with open("outputs/model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("outputs/vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)

print("\nModel and Vectorizer Saved Successfully!")

while True:

    user_message = input("\nEnter a message (or type quit): ")

    if user_message.lower() == "quit":
        break

    processed_input = preprocess_text(user_message)

    transformed_message = tfidf.transform([processed_input])

    prediction = model.predict(transformed_message)[0]

    if prediction == 1:
        print("Prediction: SPAM")
    else:
        print("Prediction: HAM")