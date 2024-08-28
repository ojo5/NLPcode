import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import random

# Download necessary NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Prepare the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Define a function to get features from a document
def get_features(document):
    words = word_tokenize(document.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return Counter(words)

# Prepare the feature sets
feature_sets = [(get_features(' '.join(doc)), category) for (doc, category) in documents]

# Split into training and testing sets
train_set, test_set = feature_sets[:1600], feature_sets[1600:]

# Train the classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test the classifier
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy:.2f}")

# Function to classify new text
def classify_text(text):
    features = get_features(text)
    return classifier.classify(features)

# Example usage
print(classify_text("This movie was amazing! I loved every minute of it."))
print(classify_text("I hated this film. It was a complete waste of time."))