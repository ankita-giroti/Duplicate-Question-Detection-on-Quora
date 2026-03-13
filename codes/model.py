import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

dataset_path = r"D:\msc_rtmnu\mini_project\mini_project_code\quora_duplicate_questions.csv"

def preprocess_text(text):
    lower_text = text.lower()
    pattern = r'[^a-zA-Z0-9\s]'
    cleaned_text = re.sub(pattern, '', lower_text)
    return cleaned_text

def train_model(dataset_path):
    dataset = pd.read_csv(dataset_path)

    # Preprocess the entire .csv file on question1 and question2 column
    dataset['question1_cleaned'] = dataset['question1'].apply(preprocess_text)
    dataset['question2_cleaned'] = dataset['question2'].apply(preprocess_text)
    
    # Apply TF-IDF to vectorized words in training set
    tfidf_vectorizer = TfidfVectorizer()

    # Create matrix of question1 and question2
    tfidf_matrix1 = tfidf_vectorizer.fit_transform(dataset['question1_cleaned'])
    tfidf_matrix2 = tfidf_vectorizer.transform(dataset['question2_cleaned'])

    # Combine the pre-processed data
    dataset['combined_cleaned'] = dataset['question1_cleaned'] + ' ' + dataset['question2_cleaned']

    x = dataset['combined_cleaned']     # features
    y = dataset['is_duplicate']         # labels

    # Split data into training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier()

    # Train accuracy
    train_accuracy = rf_model.fit(x_train, y_train)
    print('Train Accuracy: ', train_accuracy)

    # Test accuracy
    test_accuracy = rf_model.score(x_test, y_test)
    print('Test Accuracy: ', test_accuracy)

    # Compute cosine similarity between tokenized representations
    similarity_scores = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

    threshold = 0.8

    # Check if similarity scores exceed the threshold
    similar_pairs = [(dataset['question1_cleaned'][i], dataset['question2_cleaned'][j])
                     for i in range(len(dataset['question1_cleaned'])) for j in range(len(dataset['question2_cleaned']))
                     if
                     similarity_scores[i, j] > threshold]

    print("Question pairs that has same meaning:\n")

    for pair in similar_pairs:
        print(pair[0])
        print(pair[1])
        print()

    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)

    # Train the model using Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train_tfidf, y_train)

    # Evaluate the model on the test set
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    y_pred = rf_model.predict(x_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of the evaluated model is ", accuracy)

    return rf_model, tfidf_vectorizer


def predict_duplicates(questions, model, tfidf):
    preprocessed_questions = [preprocess_text(question) for question in questions]
    tfidf_matrix = tfidf.transform(preprocessed_questions)
    predictions = model.predict(tfidf_matrix)
    return "Duplicate" if predictions == 1 else "Not Duplicate"


# rf_model, tfidf_vector = train_model(dataset_path)
# print("Random Forest: ", rf_model)
# print("TFID Vector: ", tfidf_vector)