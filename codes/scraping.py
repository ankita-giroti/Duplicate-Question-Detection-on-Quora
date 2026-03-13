import requests
from bs4 import BeautifulSoup


def scrape_quora():
    url = 'https://www.quora.com/What-is-python-with-example/'
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        question_elements = soup.find_all('div', class_='question_text')
        questions = [element.get_text(strip=True) for element in question_elements]
        return questions
    else:
        print('Failed to retrieve Quora page')
        return []


def scrape_and_predict(model, vectorizer, questions):
    preprocessed_questions = [model.preprocess_text(questions) for question in questions]
    tfidf_matrix = vectorizer.transform(preprocessed_questions)
    predictions = model.predict(tfidf_matrix)
    for question, prediction in zip(questions, predictions):
        if prediction == 1:
            print(f'Question: {question} - Duplicate')
        else:
            print(f'Question: {question} - Not Duplicate')
