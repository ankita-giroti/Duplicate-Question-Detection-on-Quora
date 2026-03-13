import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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

    # Combine the pre-processed data
    dataset['combined_cleaned'] = dataset['question1_cleaned'] + ' ' + dataset['question2_cleaned']

    x = dataset[['question1_cleaned', 'question2_cleaned']]  # features
    y = dataset['is_duplicate']  # labels

    # Split data into training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Apply TF-IDF to vectorized words in training set
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the combined training set
    x_train_combined = x_train['question1_cleaned'] + ' ' + x_train['question2_cleaned']
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train_combined)

    # Train the model using Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train_tfidf, y_train)

    # Evaluate the model on the test set
    x_test_combined = x_test['question1_cleaned'] + ' ' + x_test['question2_cleaned']
    x_test_tfidf = tfidf_vectorizer.transform(x_test_combined)
    y_pred = rf_model.predict(x_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of the evaluated model is ", accuracy)

    return rf_model, tfidf_vectorizer


def predict_duplicates(question1, question2, model, tfidf):
    # Preprocess the input questions
    q1_cleaned = preprocess_text(question1)
    q2_cleaned = preprocess_text(question2)

    # Combine the cleaned questions
    combined_cleaned = [q1_cleaned + " " + q2_cleaned]

    # Transform the combined questions with the TF-IDF vectorizer
    tfidf_matrix = tfidf.transform(combined_cleaned)

    # Predict using the trained model
    prediction = model.predict(tfidf_matrix)
    return "Duplicate" if prediction[0] == 1 else "Not Duplicate"


rf_model, tfidf_vectorizer = train_model("quora_questions.csv")

question1 = input('Enter question 1: ')
question2 = input('Enter question 2: ')

result = predict_duplicates(question1, question2, rf_model, tfidf_vectorizer)
print(f"The questions are {result}")
