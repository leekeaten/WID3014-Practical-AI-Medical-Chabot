# Importing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download nltk resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    # Tokenization
    words = word_tokenize(text.lower())
    # Removing stopwords and non-alphabetic characters
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

# TF-IDF function
def tfidf(preprocessed_text, preprocessed_symptoms):
    tfidf_vectorizer = TfidfVectorizer(max_features=1500)  # You can adjust max_features based on your dataset size
    tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_symptoms).toarray()
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    return text_tfidf

# Check if input contains any recognizable symptoms
def validate_input(input, symptoms):
    # Split the preprocessed input sentence into individual words
    input_words = set(input.split())

    # Initialize a set to store all symptom words by splitting and flattening the symptoms list
    all_symptom_words = set(' '.join(symptoms).split())

    # Find the intersection of input words and symptom words
    valid_symptom_words = input_words.intersection(all_symptom_words)

    # Count the number of valid symptom words found
    valid_words_count = len(valid_symptom_words)

    if valid_words_count == 0:
        return False, 'Not Valid'
    elif valid_words_count < 5:
        return False, 'Insufficient'
    return True, ' '
