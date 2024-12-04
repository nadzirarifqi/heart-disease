import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import pickle

# Load the Heart Disease dataset
def modelling():
    st.title('Modelling')
    dataset = 'https://storage.googleapis.com/dqlab-dataset/heart_disease.csv'
    heart_data = pd.read_csv(dataset)

    # Preprocessing and cleaning
    heart_data = heart_data.dropna()
    X = heart_data.drop('target', axis=1)
    y = heart_data['target']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=9)
    X_reduced = pca.fit_transform(X_scaled)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    # Model Selection
    models = [
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('Neural Network', MLPClassifier(random_state=42, max_iter=500))
    ]

    results = []

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        })

    # Convert results to DataFrame for a tabular display
    results_df = pd.DataFrame(results)
    st.write(results_df)

    # Identify the best model
    best_model_row = results_df.loc[results_df['Accuracy'].idxmax()]
    st.write("\nBest Model Based on Accuracy:")
    st.write(best_model_row)

    # Model Tuning for Logistic Regression
    param_grid = {
        'C': [0.1, 1, 10,100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # Needed for L1 penalty
    }

    grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate the best-tuned model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with open('best_model_heart_disease.pkl','wb') as file:
        pickle.dump(best_model, file)

    st.write("\nBest Model: Logistic Regression (Tuned)")
    st.write("Best Parameters:", best_params)
    st.write(f"Accuracy: {accuracy:.2f}")
