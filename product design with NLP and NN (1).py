# import basic required librarires 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
#import data set
data=pd.read_excel('IPHONE.xlsx') 
data.head()
data.columns
# adding rateing as a next column in data 
data = data[['Review', 'Rating']]
data.head()
# we are classifying the data here to filter out the comment which have positive reviews for the product
# Rateing over 3 is considered as reviews with positive atitiude 
# features mentioned in reviews with rateing less than 3 condidered as featured needed improvment 
data['Rating'] = data['Rating'].astype(float)
for x in data['Rating']:
    def replace_values(x):
        if x > 3:
            return 'Good'
        else:
            return 'Need Improve'
    data['Rating'] = data['Rating'].apply(replace_values)

data.head()
# Convert 'rev' column to lowercase
data['Review'] = data['Review'].str.lower()
df = data
data.head()

import re
# Function to remove HTML tags and URLs
def remove_html_and_urls(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return text
    
data['Review'] = data['Review'].apply(remove_html_and_urls)
data.head()

# Function to remove special characters
def remove_special_characters(text):
    # Remove special characters using regex
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text
# Apply function to 'rev' column
data['Review'] = data['Review'].apply(remove_special_characters)
data.head()

import nltk
from nltk.corpus import stopwords
# Download stopwords data
nltk.download('stopwords')
def remove_stopwords(text):
    # Tokenize the text
    words = text.split()
    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
    # Join the filtered words back into a sentence
    filtered_text = ' '.join(filtered_words)
    return filtered_text
data['Review'] = data['Review'].apply(remove_stopwords)
data.head()

# since some features of mobile is specific with numbers we are not removing any numbers 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function for stemming and lemmatization
def stem_and_lemmatize(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Stemming
    stemmed_words = [stemmer.stem(word) for word in words]
    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(stemmed_words), ' '.join(lemmatized_words)

# Apply function to 'd' column
data['Review'], data['Review'] = zip(*data['Review'].map(stem_and_lemmatize))

data.head()

# Text data is now optimized for feature selction

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
# Fit and transform the 'Review' column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Review'])
# Convert the TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())

print(tfidf_df)

# createing tri diagram to find the features 

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

vectorizer = CountVectorizer(ngram_range=(3, 3))
X = vectorizer.fit_transform(df['Review'])

# Get trigram names
trigram_names = vectorizer.get_feature_names()
# Get trigram frequencies
trigram_frequencies = X.sum(axis=0).A1
# Create DataFrame with trigrams and frequencies
trigram_df = pd.DataFrame({'Trigram': trigram_names, 'Frequency': trigram_frequencies})

# Plot the top 20 trigrams
plt.figure(figsize=(20, 6))
plt.barh(trigram_df['Trigram'][:20], trigram_df['Frequency'][:20], color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Trigram')
plt.title('Top 20 Trigrams')
plt.gca().invert_yaxis()  # Invert y-axis to display highest frequency at the top
plt.show()

#Bow modeling for NLP
from sklearn.feature_extraction.text import CountVectorizer


# Initialize CountVectorizer for Bag-of-Words representation
vectorizer = CountVectorizer()
# Fit and transform the 'd' column
bow_matrix = vectorizer.fit_transform(data['Review'])
# Convert the BoW matrix to DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names())

print(bow_df)

pip install gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Tokenize the text in the 'd' column
tokenized_text = [word_tokenize(text) for text in data['Review']]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# Get word embeddings for each word in each sentence
word_embeddings = []
for sentence in tokenized_text:
    sentence_embeddings = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv]
    word_embeddings.append(sentence_embeddings)

print("Word embeddings for each word in each sentence:")
print(word_embeddings)

label_encoder = LabelEncoder()
data['Rating'] = label_encoder.fit_transform(data['Rating'])

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Review'])
X = tokenizer.texts_to_sequences(data['Review'])
X = pad_sequences(X)
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, data['Rating'], test_size=0.2, random_state=42)


############################
# Model with RNN
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=X.shape[1]))
model.add(LSTM(units=100))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Encode labels
label_encoder = LabelEncoder()
data['Rating'] = label_encoder.fit_transform(data['Rating'])
max_words = 1000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(data['Review'])
X = tokenizer.texts_to_sequences(data['Review'])
X = pad_sequences(X, maxlen=max_len)
X_train, X_test, y_train, y_test = train_test_split(X, data['Rating'], test_size=0.2, random_state=42)

########################################################################
# Model with CNN
embedding_dim = 100
num_filters = 64
kernel_size = 3

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(Conv1D(num_filters, kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# Train the model
batch_size = 32
epochs = 10
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)



###### Cod used for the review extraction from amazon
#### Used library BeautifulSoup


import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_amazon_reviews(url, max_pages=20):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    reviews = []
    ratings = []
    
    for page in range(1, max_pages+1):
        page_url = f"{url}?pageNumber={page}"
        response = requests.get(page_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            review_elements = soup.find_all('div', class_='a-section review aok-relative')
            for review_element in review_elements:
                review_text = review_element.find('span', class_='review-text').get_text(strip=True)
                rating = review_element.find('i', class_='review-rating').get_text(strip=True)
                reviews.append(review_text)
                ratings.append(rating)
        else:
            print(f"Failed to fetch page {page}. Status code: {response.status_code}")
            break
    
    return reviews, ratings

# Example usage
url = 'https://www.amazon.com/Apple-iPhone-5-Unlocked/product-reviews/B07ZPQWTWF'
reviews, ratings = scrape_amazon_reviews(url)

