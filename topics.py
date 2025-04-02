# This topics.py encapsulates the functions you need to do topics/prediction identification
# To train the model, please run train_model.py

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import joblib
import re
from sklearn.model_selection import ParameterGrid
from shared_state import topics_cache


"""
Using Pre-Trained Sentence Transformer:
The transformer is already trained on large datasets to produce meaningful text embeddings.
You’re leveraging this capability without modifying the model's parameters.

Training a Separate Classifier:
After generating embeddings, you're training a downstream classifier (e.g., a MultiOutputClassifier) to map embeddings to the labels (topics in your case).
Only the classifier learns from your labeled data.

The Sentence Transformer is Pretrained:
- It has already been trained on a large dataset (e.g., natural language inference, paraphrase detection) to generate high-quality embeddings that capture the semantic meaning of text.
You are using this pre-trained model "as is" to convert text into dense vector embeddings. These embeddings are fixed and are not updated during the downstream classifier training.

The Downstream Classifier is Not Pretrained:
- The downstream classifier (e.g., Logistic Regression, Random Forest, MultiOutputClassifier, etc.) starts with random or default initial parameters.
It is trained (using .fit) on your task-specific dataset, where it learns to map the embeddings to the desired output labels (e.g., topics in your case).
This training only updates the parameters of the downstream classifier, not the Sentence Transformer.
"""


### Helper functions
# Return an array of binary values indicating the presence of each topic in the list of all topics
def topics_to_vector(topics, all_topics):
    return [1 if topic.strip() in topics else 0 for topic in all_topics]


def load_topics(csv_file_path="training_data.xlsx"):
    df = pd.read_excel(csv_file_path, sheet_name="topics", engine="openpyxl")
    return df["Topics"].tolist()


def load_training_data(csv_file_path):
    df = pd.read_excel(csv_file_path, engine="openpyxl")  # Load the training data from an Excel file
    all_topics = load_topics(csv_file_path)

    def split_topics(topics):
        return topics.split("; ")

    df["topics"] = df["topics"].apply(split_topics)  # Split topics by ;
    df = df.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
    return df, all_topics


def preprocess_text(text):
    # Replace newline characters with spaces
    text = text.replace("\n", "")
    return text


### Global Variables
model = SentenceTransformer("all-mpnet-base-v2")


def tune_hyperparameters(X_train, X_test, y_train, y_test):
    param_grid = {
        "C": [0.5, 1.0, 2.0],
        "max_iter": [500, 1000, 1500, 2000],
        "class_weight": [None, "balanced"],
        "solver": ["lbfgs", "saga"],
    }
    thresholds = [0.07, 0.08, 0.09]

    best_overall_f1 = 0
    best_overall_precision = 0
    best_overall_recall = 0
    best_overall_params = None
    best_overall_threshold = None
    best_model = None

    for params in ParameterGrid(param_grid):
        base_classifier = LogisticRegression(random_state=41, **params)
        clf = MultiOutputClassifier(base_classifier)
        clf.fit(X_train, y_train)

        # Predict for a New Question
        # Interpretation of predicted_probabilities:
        # This array corresponds to the predicted probabilities for one of the labels (topics).
        # Each row in the array corresponds to a sample in the test set.
        # The first column contains the probabilities for class 0 (the negative class).
        # The second column contains the probabilities for class 1 (the positive class).
        predicted_probabilities = clf.predict_proba(X_test)

        # Try different thresholds with current parameters
        # For each pp, for each of its probability, check against threshold
        for threshold in thresholds:
            predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]
            # Currently, predicted labels is a list of numpy arrays, where each array contains the predicted labels for each topic
            # We transpose it to make it easier to get the topics for each question
            # Now each inner array is for one question, and each 1 and 0 is for each topics for that question
            transposed_labels = np.array(predicted_labels).T

            f1 = f1_score(y_test, transposed_labels, average="micro")
            precision = precision_score(y_test, transposed_labels, average="micro")
            recall = recall_score(y_test, transposed_labels, average="micro")

            if f1 > best_overall_f1:
                best_overall_f1 = f1
                best_overall_params = params
                best_overall_threshold = threshold
                best_overall_precision = precision
                best_overall_recall = recall
                best_model = clf

    return (
        best_model,
        best_overall_threshold,
        best_overall_f1,
        best_overall_precision,
        best_overall_recall,
        best_overall_params,
    )


### Function to train the classification model
def train_classifier():
    training_data, all_topics = load_training_data("training_data.xlsx")

    # Step 1: Processing Data
    # X_text is the questions for each training data
    X_text = [preprocess_text(item["question"]) for item in training_data]
    # Y is the binary vector representation of the topics for each question (2D Numpy Array)
    Y = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

    # Step 2: Encode questions using SentenceTransformer
    X_embedding = model.encode(X_text)

    # Step 3: Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_embedding, Y, test_size=0.2, random_state=41)

    # # Step 4: Scale the encoded embeddings
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # # Step 5: Dimensionality Reduction using PCA
    pca = PCA(
        n_components=0.95
    )  # Retain 95% of variance Aim: to reduce the number of features while retaining the variance
    X_reduced = pca.fit_transform(X_scaled)
    X_test_reduced = pca.transform(X_test_scaled)

    # Hyperparameter tuning
    best_model, threshold, f1, precision, recall, params = tune_hyperparameters(
        X_reduced, X_test_reduced, y_train, y_test
    )

    print(f"\nBest overall configuration:")
    print(f"Parameters: {params}")
    print(f"Threshold: {threshold}")
    print(f"F1 score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Save the trained model
    joblib.dump(best_model, "multi_label_classifier.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca.pkl")
    joblib.dump(threshold, "threshold.pkl")

    print("\nTraining Topics Prediction Model Complete and Saved")


# Function to predict topics for new questions
def predict_topics(new_questions):
    try:
        # Load the trained model and threshold
        multi_label_classifier = joblib.load("multi_label_classifier.pkl")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
        threshold = joblib.load("threshold.pkl")
    except FileNotFoundError:
        raise FileNotFoundError(
            'Topic Identification Model has not been trained. Please run the command "python train_model.py" first.'
        )

    # Encode new questions using the same SentenceTransformer model
    new_embeddings = model.encode(new_questions)

    # Scale the encoded embeddings using the already fitted scaler
    scaled_embedding = scaler.transform(new_embeddings)

    # Reduce dimensionality using the already fitted PCA
    reduced_embedding = pca.transform(scaled_embedding)

    # Predict probabilities
    predicted_probabilities = multi_label_classifier.predict_proba(reduced_embedding)

    # Assign topics based on threshold
    predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]
    transposed_labels = np.array(predicted_labels).T

    # Get topic names for each question
    question_predictions = []
    for q in transposed_labels:
        predicted_topics = [topics_cache[i] for i, label in enumerate(q) if label == 1]
        question_predictions.append(predicted_topics)

    return question_predictions
