import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
app = Flask(__name__)

CSV_FILE_PATH = "diabetes.csv"  # Change this path to the actual CSV file

def preprocess_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna('', inplace=True)

    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df

def evaluate_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    return {
        'accuracy': round(np.mean(acc_scores) * 100, 2),
        'precision': round(np.mean(prec_scores) * 100, 2),
        'recall': round(np.mean(rec_scores) * 100, 2),
        'f1_score': round(np.mean(f1_scores) * 100, 2)
    }

@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        X, y, processed_df = preprocess_data(df)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Support Vector Machine': SVC(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }

        results = []
        for model_name, model in models.items():
            metrics = evaluate_model(model, X, y)
            results.append({'model': model_name, **metrics})

        best_model = max(results, key=lambda x: x['accuracy'])

        response = {
            'results': results,
            'best_model': best_model,
            'dataset_info': {
                'shape': processed_df.shape,
                'missing_values': processed_df.isnull().sum().to_dict(),
                'data_types': processed_df.dtypes.apply(str).to_dict()
            }
        }

        # Manually setting headers to allow all origins
        json_response = jsonify(response)
        json_response.headers.add('Access-Control-Allow-Origin', '*')
        json_response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        json_response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')

        return json_response

    except Exception as e:
        error_response = jsonify({'error': str(e)})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))  # Use Render's assigned port or default to 5000
    app.run(host='0.0.0.0', port=port)
