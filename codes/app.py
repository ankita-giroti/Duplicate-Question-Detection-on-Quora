from flask import Flask, request, jsonify
from threading import Thread
import model
import scraping

app = Flask(__name__)


@app.route('/detect_duplicates', methods=['POST'])
def detect_duplicates():
    if request.method == 'POST':
        data = request.json
        new_questions = data['questions']
        predictions = model.predict_duplicates(new_questions)
        return jsonify({'predictions': predictions.tolist()})


if __name__ == "__main__":
    rf_model, tfidf_vectorizer = model.train_model("quora_dataset.csv")
    thread = Thread(target=scraping.scrape_and_predict, args=(rf_model, tfidf_vectorizer))
    thread.start()
    app.run(debug=True)
