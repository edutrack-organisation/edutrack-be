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
from topics_data import all_topics
import joblib
import re
from sklearn.model_selection import ParameterGrid


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


def load_training_data(csv_file_path):
    df = pd.read_excel(csv_file_path, engine="openpyxl")  # Load the training data from an Excel file

    def split_topics(topics):
        return topics.split("; ")

    df["topics"] = df["topics"].apply(split_topics)  # Split topics by ;
    df = df.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
    return df


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
    training_data = load_training_data("training_data.xlsx")

    # Step 1: Processing Data
    # X_text is the questions for each training data
    X_text = [preprocess_text(item["question"]) for item in training_data]
    # Y is the binary vector representation of the topics for each question (2D Numpy Array)
    Y = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

    # Step 2: Encode questions using SentenceTransformer
    X_embedding = model.encode(X_text)

    # Step 3: Split the data into training and test sets
    # # X_test and X train is the embedding of each question, where the first dimension is the question and the second dimension is the embedding
    # X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=41)

    X_train, X_test, y_train, y_test = train_test_split(X_embedding, Y, test_size=0.2, random_state=41)

    # # Step 4: Scale the encoded embeddings
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # # Step 4: Dimensionality Reduction using PCA
    pca = PCA(
        n_components=0.95
    )  # Retain 95% of variance Aim: to reduce the number of features while retaining the variance
    X_reduced = pca.fit_transform(X_scaled)
    X_test_reduced = pca.transform(X_test_scaled)

    # Hyperparamter tuning
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
    # Save the trained model
    joblib.dump(best_model, "multi_label_classifier.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca.pkl")
    joblib.dump(threshold, "threshold.pkl")

    print("\nTraining Topics Prediction Model Complete and Saved")


# Function to predict topics for new questions
def predict_topics(new_questions):
    # Load the trained model and threshold
    multi_label_classifier = joblib.load("multi_label_classifier.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    threshold = joblib.load("threshold.pkl")

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
        predicted_topics = [all_topics[i] for i, label in enumerate(q) if label == 1]
        question_predictions.append(predicted_topics)

    return question_predictions


# This topics.py encapsulates the functions you need to do topics/prediction identification
# To train the model, please run train_model.py

# v1 tuning
# import numpy as np
# import pandas as pd
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestCentroid
# from sentence_transformers import SentenceTransformer
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import f1_score, recall_score, precision_score
# from sklearn.model_selection import train_test_split
# from topics_data import all_topics
# import joblib
# import re
# from xgboost import XGBClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import ParameterGrid

# # from sklearn.svm import LinearSVC
# # from sklearn.calibration import CalibratedClassifierCV


# """
# Using Pre-Trained Sentence Transformer:
# The transformer is already trained on large datasets to produce meaningful text embeddings.
# You’re leveraging this capability without modifying the model's parameters.

# Training a Separate Classifier:
# After generating embeddings, you're training a downstream classifier (e.g., a MultiOutputClassifier) to map embeddings to the labels (topics in your case).
# Only the classifier learns from your labeled data.

# The Sentence Transformer is Pretrained:
# - It has already been trained on a large dataset (e.g., natural language inference, paraphrase detection) to generate high-quality embeddings that capture the semantic meaning of text.
# You are using this pre-trained model "as is" to convert text into dense vector embeddings. These embeddings are fixed and are not updated during the downstream classifier training.

# The Downstream Classifier is Not Pretrained:
# - The downstream classifier (e.g., Logistic Regression, Random Forest, MultiOutputClassifier, etc.) starts with random or default initial parameters.
# It is trained (using .fit) on your task-specific dataset, where it learns to map the embeddings to the desired output labels (e.g., topics in your case).
# This training only updates the parameters of the downstream classifier, not the Sentence Transformer.
# """


# ### Helper functions
# # Return an array of binary values indicating the presence of each topic in the list of all topics
# def topics_to_vector(topics, all_topics):
#     return [1 if topic.strip() in topics else 0 for topic in all_topics]


# def load_training_data(csv_file_path):
#     df = pd.read_excel(csv_file_path, engine="openpyxl")  # Load the training data from an Excel file

#     def split_topics(topics):
#         return topics.split("; ")

#     df["topics"] = df["topics"].apply(split_topics)  # Split topics by ;
#     df = df.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
#     return df


# def preprocess_text(text):
#     # Replace newline characters with spaces
#     # NOTE: May or may not require the line below, need experiment
#     text = text.replace("\n", "")
#     return text


# # Add GridSearchCV for hyperparameter tuning
# from sklearn.model_selection import GridSearchCV


# ### Global Variables
# model = SentenceTransformer("all-mpnet-base-v2")


# ### Function to train the classification model
# def train_classifier():
#     training_data = load_training_data("training_data.xlsx")

#     # print(training_data[0])

#     # Step 1: Processing Data
#     # X_text is the questions for each training data
#     X_text = [preprocess_text(item["question"]) for item in training_data]
#     # Y is the binary vector representation of the topics for each question (2D Numpy Array)
#     Y = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

#     # Step 2: Encode questions using SentenceTransformer
#     X_embedding = model.encode(X_text)

#     # Step 3: Split the data into training and test sets
#     # # X_test and X train is the embedding of each question, where the first dimension is the question and the second dimension is the embedding
#     # X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=41)

#     X_train, X_test, y_train, y_test = train_test_split(X_embedding, Y, test_size=0.2, random_state=41)

#     # # Step 4: Scale the encoded embeddings
#     scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
#     X_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # # Step 4: Dimensionality Reduction using PCA
#     pca = PCA(
#         n_components=0.95
#     )  # Retain 95% of variance Aim: to reduce the number of features while retaining the variance
#     X_reduced = pca.fit_transform(X_scaled)

#     X_train = X_reduced
#     X_test = pca.transform(X_test_scaled)

#     # # DEBUG: debugging hash map
#     # hash_map = {}
#     # for i in range(len(y_train)):
#     #     for j in range(len(y_train[0])):
#     #         if j not in hash_map:
#     #             hash_map[j] = 0
#     #         elif y_train[i][j] == 1:
#     #             hash_map[j] += 1
#     # print(hash_map)

#     # # Step 4: Train a Multi-Label Classifier

#     # # #DEBUG: For debugging
#     # # np.set_printoptions(threshold=np.inf)
#     # # print(y_train)

#     # # NOTE: You need to have the sufficient data and you need to have at least one row for each topics in all_topics for y_train

#     # base_classifier = LogisticRegression(
#     #     max_iter=1000,
#     # )

#     # # base_classifier = RandomForestClassifier(n_estimators=270, random_state=42)
#     # multi_label_classifier = MultiOutputClassifier(base_classifier)
#     # multi_label_classifier.fit(X_train, y_train)  # trains the multi_label_classifier

#     # Define parameters including threshold
#     # param_grid = {
#     #     "estimator__C": [0.1, 0.5, 1.0, 2.0],
#     #     "estimator__max_iter": [500, 1000, 1500],
#     #     "estimator__class_weight": [None, "balanced"],
#     #     "estimator__solver": ["lbfgs", "saga"],
#     #     "estimator__penalty": ["l2", None],
#     # }

#     # ... existing code ...
#     param_grid = {
#         "C": [0.5, 1.0, 2.0],
#         "max_iter": [500, 1000, 1500, 2000],
#         "class_weight": [None, "balanced"],
#         "solver": ["lbfgs", "saga"],
#     }

#     thresholds = [0.07, 0.08, 0.09]  # Focus around our known good
#     best_overall_f1 = 0
#     best_overall_precision = 0
#     best_overall_accuracy = 0
#     best_overall_recall = 0
#     best_overall_params = None
#     best_overall_threshold = None
#     best_model = None

#     for params in ParameterGrid(param_grid):
#         # Configure classifier with current parameters
#         base_classifier = LogisticRegression(random_state=41, **params)
#         clf = MultiOutputClassifier(base_classifier)
#         clf.fit(X_train, y_train)

#         # Try different thresholds with current parameters
#         predicted_probabilities = clf.predict_proba(X_test)
#         for threshold in thresholds:
#             predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]
#             transposed_labels = np.array(predicted_labels).T
#             f1 = f1_score(y_test, transposed_labels, average="micro")
#             precision = precision_score(y_test, transposed_labels, average="micro")
#             accuracy = (y_test == transposed_labels).mean()
#             recall = recall_score(y_test, transposed_labels, average="micro")

#             if f1 > best_overall_f1:
#                 best_overall_f1 = f1
#                 best_overall_params = params
#                 best_overall_threshold = threshold
#                 best_overall_precision = precision
#                 best_overall_accuracy = accuracy
#                 best_model = clf
#                 best_overall_recall = recall

#     print(f"\nBest overall configuration:")
#     print(f"Parameters: {best_overall_params}")
#     print(f"Threshold: {best_overall_threshold}")
#     print(f"F1 score: {best_overall_f1}")
#     print(f"Precision: {best_overall_precision}")
#     print(f"Accuracy: {best_overall_accuracy}")
#     print(f"Recall: {best_overall_recall}")

#     multi_label_classifier = best_model
#     threshold = best_overall_threshold

#     # Save the trained model
#     joblib.dump(multi_label_classifier, "multi_label_classifier.pkl")
#     joblib.dump(scaler, "scaler.pkl")
#     joblib.dump(pca, "pca.pkl")

#     # Step 5: Predict for a New Question
#     # Intepretation of predicted_probabilities
#     # This array corresponds to the predicted probabilities for one of the labels (topics).
#     # Each row in the array corresponds to a sample in the test set.
#     # The first column contains the probabilities for class 0 (the negative class).
#     # The second column contains the probabilities for class 1 (the positive class).
#     predicted_probabilities = multi_label_classifier.predict_proba(X_test)

#     # Step 6: Assign Topics Based on Threshold
#     threshold = 0.08  # Adjust this threshold as needed #TODO: need a better way to determine threshold
#     # for each pp, for each of it's probability, check the index 0 against threshold
#     predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]

#     # Currently, predicted labels is a list of numpy arrays, where each array contains the predicted labels for each topic. e.g. [array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0])]. Each array is for the topics, while the 1 and 0 indicates FOR each questions FOR THIS TOPIC
#     # We transposed it to make it easier to get the topics for each question
#     # Now each inner array is for one question, and each 1 and 0 is for each topics for that QUESTION
#     transposed_labels = np.array(predicted_labels).T

#     # Step 7: Get topic names for each question
#     question_predictions = []
#     for q in transposed_labels:
#         predicted_topics = [all_topics[i] for i, label in enumerate(q) if label == 1]
#         question_predictions.append(predicted_topics)

#     f1_scores = f1_score(y_test, transposed_labels, average="micro")
#     recall_scores = recall_score(y_test, transposed_labels, average="micro")
#     precision_scores = precision_score(y_test, transposed_labels, average="micro")

#     # TODO: mention the steps taken to improve in r
#     print("Training Topics Prediction Model Complete and Saved")
#     print(f"F1-Score for Multi-Label Topic Classification: {f1_scores}")
#     print(f"Recall-Score for Multi-Label Topic Classification: {recall_scores}")
#     print(f"Precision-Score for Multi-Label Topic Classification: {precision_scores}")


# # Function to predict topics for new questions
# # TODO: Good to abstract into pipeline?
# def predict_topics(new_questions):
#     # Load the trained model
#     multi_label_classifier = joblib.load("multi_label_classifier.pkl")
#     scaler = joblib.load("scaler.pkl")
#     pca = joblib.load("pca.pkl")

#     # Encode new questions using the same SentenceTransformer model
#     new_embeddings = model.encode(new_questions)

#     # Scale the encoded embeddings using the already fitted scaler
#     scaled_embedding = scaler.transform(new_embeddings)

#     # Reduce dimensionality using the already fitted PCA
#     reduced_embedding = pca.transform(scaled_embedding)

#     # Predict probabilities
#     predicted_probabilities = multi_label_classifier.predict_proba(reduced_embedding)

#     # Assign topics based on threshold
#     threshold = 0.08
#     predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]
#     transposed_labels = np.array(predicted_labels).T

#     # Get topic names for each question
#     question_predictions = []
#     for q in transposed_labels:
#         predicted_topics = [all_topics[i] for i, label in enumerate(q) if label == 1]
#         question_predictions.append(predicted_topics)

#     return question_predictions


# before hyperparameter tune

# # This topics.py encapsulates the functions you need to do topics/prediction identification
# # To train the model, please run train_model.py

# import numpy as np
# import pandas as pd
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestCentroid
# from sentence_transformers import SentenceTransformer
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import f1_score, recall_score, precision_score
# from sklearn.model_selection import train_test_split
# from topics_data import all_topics
# import joblib
# import re
# from xgboost import XGBClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier

# # from sklearn.svm import LinearSVC
# # from sklearn.calibration import CalibratedClassifierCV


# """
# Using Pre-Trained Sentence Transformer:
# The transformer is already trained on large datasets to produce meaningful text embeddings.
# You’re leveraging this capability without modifying the model's parameters.

# Training a Separate Classifier:
# After generating embeddings, you're training a downstream classifier (e.g., a MultiOutputClassifier) to map embeddings to the labels (topics in your case).
# Only the classifier learns from your labeled data.

# The Sentence Transformer is Pretrained:
# - It has already been trained on a large dataset (e.g., natural language inference, paraphrase detection) to generate high-quality embeddings that capture the semantic meaning of text.
# You are using this pre-trained model "as is" to convert text into dense vector embeddings. These embeddings are fixed and are not updated during the downstream classifier training.

# The Downstream Classifier is Not Pretrained:
# - The downstream classifier (e.g., Logistic Regression, Random Forest, MultiOutputClassifier, etc.) starts with random or default initial parameters.
# It is trained (using .fit) on your task-specific dataset, where it learns to map the embeddings to the desired output labels (e.g., topics in your case).
# This training only updates the parameters of the downstream classifier, not the Sentence Transformer.
# """


# ### Helper functions
# # Return an array of binary values indicating the presence of each topic in the list of all topics
# def topics_to_vector(topics, all_topics):
#     return [1 if topic.strip() in topics else 0 for topic in all_topics]


# def load_training_data(csv_file_path):
#     df = pd.read_excel(csv_file_path, engine="openpyxl")  # Load the training data from an Excel file

#     def split_topics(topics):
#         return topics.split("; ")

#     df["topics"] = df["topics"].apply(split_topics)  # Split topics by ;
#     df = df.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
#     return df


# def preprocess_text(text):
#     # Replace newline characters with spaces
#     # NOTE: May or may not require the line below, need experiment
#     text = text.replace("\n", "")
#     return text


# ### Global Variables
# model = SentenceTransformer("all-mpnet-base-v2")


# ### Function to train the classification model
# def train_classifier():
#     training_data = load_training_data("training_data.xlsx")

#     # print(training_data[0])

#     # Step 1: Processing Data
#     # X_text is the questions for each training data
#     X_text = [preprocess_text(item["question"]) for item in training_data]
#     # Y is the binary vector representation of the topics for each question (2D Numpy Array)
#     Y = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

#     # Step 2: Encode questions using SentenceTransformer
#     X_embedding = model.encode(X_text)

#     X_train, X_test, y_train, y_test = train_test_split(X_embedding, Y, test_size=0.2, random_state=41)

#     # # Step 3: Scale the encoded embeddings
#     scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
#     X_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # # X_scaled = X_embedding / np.linalg.norm(X_embedding, axis=1)[:, np.newaxis]

#     # # Step 4: Dimensionality Reduction using PCA
#     pca = PCA(
#         n_components=0.95
#     )  # Retain 95% of variance Aim: to reduce the number of features while retaining the variance
#     X_reduced = pca.fit_transform(X_scaled)

#     X_train = X_reduced
#     X_test = pca.transform(X_test_scaled)

#     # X_train = X_scaled
#     # X_test = X_test_scaled

#     # Step 3: Split the data into training and test sets
#     # # X_test and X train is the embedding of each question, where the first dimension is the question and the second dimension is the embedding
#     # X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=41)

#     # DEBUG: debugging hash map
#     hash_map = {}
#     for i in range(len(y_train)):
#         for j in range(len(y_train[0])):
#             if j not in hash_map:
#                 hash_map[j] = 0
#             elif y_train[i][j] == 1:
#                 hash_map[j] += 1
#     print(hash_map)
#     # Step 4: Train a Multi-Label Classifier

#     # #DEBUG: For debugging
#     # np.set_printoptions(threshold=np.inf)
#     # print(y_train)

#     # NOTE: You need to have the sufficient data and you need to have at least one row for each topics in all_topics for y_train

#     # base_classifier = KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
#     # base_classifier = LinearSVC(max_iter=1000, dual=False)
#     # base_svc = LinearSVC(
#     #     max_iter=1000, dual=False, class_weight="balanced", C=1.0  # Handle class imbalance
#     # )  # Increase regularization)
#     # base_classifier = CalibratedClassifierCV(base_svc, n_jobs=-1)  # Reduce from default 5 due to small class sizes

#     # base_classifier = RandomForestClassifier(
#     #     n_estimators=500,
#     #     # max_depth=None,
#     #     # min_samples_split=2,
#     #     # min_samples_leaf=1,
#     #     max_features="sqrt",
#     #     class_weight="balanced",
#     #     # bootstrap=True,
#     #     random_state=41,
#     #     n_jobs=-1,
#     # )

#     # base_classifier = XGBClassifier(
#     #     n_estimators=500,
#     #     max_depth=6,
#     #     learning_rate=0.1,
#     #     subsample=0.8,
#     #     colsample_bytree=0.8,
#     #     scale_pos_weight=1,  # Helps with class imbalance
#     #     random_state=41,
#     #     n_jobs=-1,
#     #     use_label_encoder=False,
#     #     eval_metric="mlogloss",
#     # )

#     base_classifier = LogisticRegression(
#         max_iter=1000,
#         # Increase iterations for better convergence
#         # class_weight="balanced",  # Handle class imbalance
#     )  # For reproducibility)  # Use all CPU cores)
#     # base_classifier = MLPClassifier(
#     #     hidden_layer_sizes=(384, 192, 96),  # Gradually reduce dimensions
#     #     activation="relu",
#     #     solver="adam",
#     #     alpha=0.0001,  # L2 regularization
#     #     batch_size="auto",
#     #     learning_rate="adaptive",
#     #     max_iter=1000,
#     #     early_stopping=True,  # Prevent overfitting
#     #     validation_fraction=0.1,
#     #     n_iter_no_change=10,
#     #     random_state=42,
#     # )

#     # base_classifier = RandomForestClassifier(n_estimators=270, random_state=42)
#     multi_label_classifier = MultiOutputClassifier(base_classifier)
#     multi_label_classifier.fit(X_train, y_train)  # trains the multi_label_classifier

#     # Save the trained model
#     joblib.dump(multi_label_classifier, "multi_label_classifier.pkl")
#     joblib.dump(scaler, "scaler.pkl")
#     joblib.dump(pca, "pca.pkl")

#     # Step 5: Predict for a New Question
#     # Intepretation of predicted_probabilities
#     # This array corresponds to the predicted probabilities for one of the labels (topics).
#     # Each row in the array corresponds to a sample in the test set.
#     # The first column contains the probabilities for class 0 (the negative class).
#     # The second column contains the probabilities for class 1 (the positive class).
#     predicted_probabilities = multi_label_classifier.predict_proba(X_test)

#     # Step 6: Assign Topics Based on Threshold
#     threshold = 0.08  # Adjust this threshold as needed #TODO: need a better way to determine threshold
#     # for each pp, for each of it's probability, check the index 0 against threshold
#     predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]

#     # Currently, predicted labels is a list of numpy arrays, where each array contains the predicted labels for each topic. e.g. [array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0])]. Each array is for the topics, while the 1 and 0 indicates FOR each questions FOR THIS TOPIC
#     # We transposed it to make it easier to get the topics for each question
#     # Now each inner array is for one question, and each 1 and 0 is for each topics for that QUESTION
#     transposed_labels = np.array(predicted_labels).T

#     # Step 7: Get topic names for each question
#     question_predictions = []
#     for q in transposed_labels:
#         predicted_topics = [all_topics[i] for i, label in enumerate(q) if label == 1]
#         question_predictions.append(predicted_topics)

#     f1_scores = f1_score(y_test, transposed_labels, average="micro")
#     recall_scores = recall_score(y_test, transposed_labels, average="micro")
#     precision_scores = precision_score(y_test, transposed_labels, average="micro")

#     # TODO: mention the steps taken to improve in r
#     print("Training Topics Prediction Model Complete and Saved")
#     print(f"F1-Score for Multi-Label Topic Classification: {f1_scores}")
#     print(f"Recall-Score for Multi-Label Topic Classification: {recall_scores}")
#     print(f"Precision-Score for Multi-Label Topic Classification: {precision_scores}")


# # Function to predict topics for new questions
# # TODO: Good to abstract into pipeline?
# def predict_topics(new_questions):
#     # Load the trained model
#     multi_label_classifier = joblib.load("multi_label_classifier.pkl")
#     scaler = joblib.load("scaler.pkl")
#     pca = joblib.load("pca.pkl")

#     # Encode new questions using the same SentenceTransformer model
#     new_embeddings = model.encode(new_questions)

#     # Scale the encoded embeddings using the already fitted scaler
#     scaled_embedding = scaler.transform(new_embeddings)

#     # Reduce dimensionality using the already fitted PCA
#     reduced_embedding = pca.transform(scaled_embedding)

#     # Predict probabilities
#     predicted_probabilities = multi_label_classifier.predict_proba(reduced_embedding)

#     # Assign topics based on threshold
#     threshold = 0.08
#     predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]
#     transposed_labels = np.array(predicted_labels).T

#     # Get topic names for each question
#     question_predictions = []
#     for q in transposed_labels:
#         predicted_topics = [all_topics[i] for i, label in enumerate(q) if label == 1]
#         question_predictions.append(predicted_topics)

#     return question_predictions
