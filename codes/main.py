import model

def main():
    rf_model, tfidf_vectorizer = model.train_model("quora_dataset.csv")

    question1 = input('Enter question 1: ')
    question2 = input('Enter question 2: ')

    question_pair = [question1, question2]
    predictions = model.predict_duplicates(question_pair, rf_model, tfidf_vectorizer)

    print(f"The questions are {predictions}")


if __name__ == "__main__":
    main()