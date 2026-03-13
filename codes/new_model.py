import pandas as pd
from sklearn.model_selection import train_test_split

ds = pd.read_csv('quora_dataset.csv')

# print("Shape: ", ds.shape)
#
# print("Head of dataframe: ", ds.head())
#
# print("Columns: ", ds.columns)

X = ds['question1']
y = ds['question2']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=0.25, shuffle=True)

print("X_train")
print(X_train.head())
print(X_train.shape)

print("\n")

print("X_test")
print(X_test.head())
print(X_test.shape)

print("\n")

print("y_train")
print(y_train.head())
print(y_train.shape)

print("\n")

print("y_test")
print(y_test.head())
print(y_test.shape)