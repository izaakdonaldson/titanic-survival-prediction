import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from preprocess import get_X_y

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

#     --- BEST MODELS ---
# Random Forest:       0.822
# XGBoost:             0.818
# Logistic Regression: 0.815

# --- Decision Tree ---
# Accuracy average after 50 runs
# 0.7781005586592181
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
# 0.7815642458100558
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
# 0.8146368715083798
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
# 0.8218994413407821
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

# --- K Nearest Neighbors ---
# Accuracy average after 50 runs
# 0.7230167597765361
def run_k_neighbors():
    from sklearn.neighbors import KNeighborsClassifier
    X, y = get_X_y(train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_prediction = knn.predict(X_test)

    print("--- Decision Tree ---")
    print("Accuracy:", accuracy_score(y_test, y_prediction))
    print("Confusion_Matrix:\n", confusion_matrix(y_test, y_prediction))
    print("Classification Report:\n", classification_report(y_test, y_prediction))

# --- Support Vector Classification ---
# Accuracy average after 50 runs
# 0.6772067039106147
def run_support_vector():
    from sklearn.svm import SVC
    X, y = get_X_y(train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svc = SVC()
    svc.fit(X_train, y_train)
    y_prediction = svc.predict(X_test)

    print("--- Decision Tree ---")
    print("Accuracy:", accuracy_score(y_test, y_prediction))
    print("Confusion_Matrix:\n", confusion_matrix(y_test, y_prediction))
    print("Classification Report:\n", classification_report(y_test, y_prediction))

# --- XGBoost ---
# Accuracy average after 50 runs
# 0.8184357541899435
def run_xgboost():
    from xgboost import XGBClassifier

    X, y = get_X_y(train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_prediction = xgb.predict(X_test)

    print("--- Decision Tree ---")
    print("Accuracy:", accuracy_score(y_test, y_prediction))
    print("Confusion_Matrix:\n", confusion_matrix(y_test, y_prediction))
    print("Classification Report:\n", classification_report(y_test, y_prediction))


# --- Rule-based classification ---
# Random
# Accuracy average after 50 runs 
# 0.616161616161616
def run_random():
    correct = 0
    for j in train['Survived']:
        rand_state = np.random.randint(0, 1)
        if rand_state == j:
            correct = correct + 1
    accuracy = correct / len(train['Survived'])
    return accuracy

# --- Rule-based classification ---
# Random
# Accuracy average after 50 runs 
# 0.7867564534231201
def run_woman_survived():
    correct = ((train['Sex'] == 'male') & (train['Survived'] == 0)) | ((train['Sex'] == 'female') & (train['Survived'] == 1))
    accuracy = correct.sum() / len(train['Survived'])
    return accuracy

# Used for running each model 50 times
'''acc = 0
for i in range(50):
    #rand_state = np.random.randint(0, 10000)
    acc = acc + run_woman_survived()
print(acc/50)'''

def kaggle_run_random_forest():
    from sklearn.linear_model import LogisticRegression
    passenger_ids = test['PassengerId']

    X_train, y_train = get_X_y(train)
    X_test, _ = get_X_y(test)

    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train, y_train)
    y_prediction = lr.predict(X_test)

    submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': y_prediction.astype(int)
    })
    submission.to_csv('submission.csv', index=False)

def kaggle_run_XGBoost():
    from xgboost import XGBClassifier
    passenger_ids = test['PassengerId']

    X_train, y_train = get_X_y(train)
    X_test, _ = get_X_y(test)

    lr = XGBClassifier(solver='liblinear')
    lr.fit(X_train, y_train)
    y_prediction = lr.predict(X_test)

    submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': y_prediction.astype(int)
    })
    submission.to_csv('submission.csv', index=False)

def kaggle_run_woman_survived():
    passenger_ids = test['PassengerId']
    predictions = (test['Sex'] == 'female').astype(int)
    submission = pd.DataFrame({
        'PassengerId': passenger_ids.astype(int),
        'Survived': predictions
    })
    submission.to_csv('submission.csv', index=False)

kaggle_run_woman_survived()
