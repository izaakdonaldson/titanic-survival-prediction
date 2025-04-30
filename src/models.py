

''' 
-- Models to impliment--
1. Decision Tree
2. Naive Bayes (GaussianNB)
3. Logistic Regression
4. Random Forest
5. KNN
6. SVM
7. XGBoost

-- Manual Classification for baseline--
Rule-based classification
Random
Majority Class (Always died)
Stratified Random (60% died 40% survived)


***
from sklearn.metrics import accuracy_score
accuracy_score(df['Survived'], df['manual_pred'])  # or whatever your column is
***
'''


import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from preprocess import get_X_y

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# --- Decision Tree ---
# Accuracy average after 50 runs
# 0.7865921787709499
def run_decision_tree():
    from sklearn.tree import DecisionTreeClassifier

    X, y = get_X_y(train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_prediction = dtc.predict(X_test)

    print("--- Decision Tree ---")
    print("Accuracy:", accuracy_score(y_test, y_prediction))
    print("Confusion_Matrix:\n", confusion_matrix(y_test, y_prediction))
    print("Classification Report:\n", classification_report(y_test, y_prediction))


# --- Naive Bayes ---
# Accuracy average after 50 runs
# 0.7888268156424582
def run_naive_bayes():
    from sklearn.naive_bayes import GaussianNB

    X, y = get_X_y(train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_prediction = gnb.predict(X_test)

    print("--- Decision Tree ---")
    print("Accuracy:", accuracy_score(y_test, y_prediction))
    print("Confusion_Matrix:\n", confusion_matrix(y_test, y_prediction))
    print("Classification Report:\n", classification_report(y_test, y_prediction))

# --- Logistic Regression ---
# Accuracy average after 50 runs
# 0.8215642458100558
def run_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    X, y = get_X_y(train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train, y_train)
    y_prediction = lr.predict(X_test)

    print("--- Decision Tree ---")
    print("Accuracy:", accuracy_score(y_test, y_prediction))
    print("Confusion_Matrix:\n", confusion_matrix(y_test, y_prediction))
    print("Classification Report:\n", classification_report(y_test, y_prediction))

# --- Random Forest ---
# Accuracy average after 50 runs
# 0.8215642458100558
def run_random_forest():
    from sklearn.linear_model import LogisticRegression
    X, y = get_X_y(train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train, y_train)
    y_prediction = lr.predict(X_test)

    print("--- Decision Tree ---")
    print("Accuracy:", accuracy_score(y_test, y_prediction))
    print("Confusion_Matrix:\n", confusion_matrix(y_test, y_prediction))
    print("Classification Report:\n", classification_report(y_test, y_prediction))









'''acc = 0
for i in range(50):
    acc = acc + run_logistic_regression(i)

print(acc/50)'''