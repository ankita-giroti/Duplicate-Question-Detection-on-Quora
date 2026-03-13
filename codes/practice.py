import pandas as pd

df = pd.read_csv("quora_dataset.csv")

print(df.shape)
print(df.head())
print()
print(df.tail())




# ********* TRAIN MODEL METHOD *************

# # Combine the pre-processed data
# # dataset['combined_cleaned'] = dataset['question1_cleaned'] + ' ' + dataset['question2_cleaned']
#
# x = dataset[['question1_cleaned', 'question2_cleaned']]
# y = dataset['is_duplicate']
#
# # Split data into training and testing dataset
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# # Apply TF-IDF to vectorized words in training set
# tfidf_vectorizer = TfidfVectorizer()
#
# # Create matrix of question1 and question2
# tfidf_matrix1 = tfidf_vectorizer.fit_transform(x_train['question1_cleaned'])
# tfidf_matrix2 = tfidf_vectorizer.transform(x_train['question2_cleaned'])
#
# threshold = 0.8
#
# # Check if similarity scores exceed the threshold
# similar_pairs = [(dataset['question1_cleaned'][i], dataset['question2_cleaned'][j])
#                  for i in range(len(dataset['question1_cleaned'])) for j in
#                  range(len(dataset['question2_cleaned'])) if similarity_scores[i, j] > threshold]
#
# print("Question pairs that has same meaning:\n")
#
# for pair in similar_pairs:
#     print(pair[0])
#     print(pair[1])
#     print()
#
# x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
#
# # Train the model using Random Forest classifier
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#
# # Transform the test data
# tfidf_matrix1_test = tfidf_vectorizer.transform(x_test['question1_cleaned'])
# tfidf_matrix2_test = tfidf_vectorizer.transform(x_test['question2_cleaned'])
#
# # Compute cosine similarity between question pairs in the test set
# similarity_scores_test = cosine_similarity(tfidf_matrix1_test, tfidf_matrix2_test)
# X_test_sim = similarity_scores_test.diagonal().reshape(-1, 1)
#
# # Evaluate the model on the test set
# x_test_tfidf = tfidf_vectorizer.transform(X_test_sim)
# y_predict = rf_model.predict(x_test_tfidf)
#
# accuracy = accuracy_score(y_test, y_predict)
# print("Accuracy of the evaluated model is ", accuracy)
#
# # print("Tfidf vectorizor: ", tfidf_vectorizer)
# # print("random forest model: ", rf_model)
#
# return rf_model, tfidf_vectorizer
