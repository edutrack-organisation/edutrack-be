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

'''
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
'''

### Helper functions
# Return an array of binary values indicating the presence of each topic in the list of all topics
def topics_to_vector(topics, all_topics):
    return [1 if topic.strip() in topics else 0 for topic in all_topics]

def load_training_data(csv_file_path):
    df = pd.read_excel(csv_file_path, engine='openpyxl')  # Load the training data from an Excel file

    def split_topics(topics):
        return topics.split("; ")

    df["topics"] = df["topics"].apply(split_topics)   # Split topics by ;
    df = df.to_dict(orient='records')    # Convert DataFrame to a list of dictionaries
    return df

def preprocess_text(text):
    # Replace newline characters with spaces
    # NOTE: May or may not require the line below, need experiment
    text = text.replace('\n', '')
    return text

### Global Variables
model = SentenceTransformer('all-MiniLM-L6-v2')

### Function to train the classification model
def train_classifier():
    training_data = load_training_data('training_data.xlsx')

    # print(training_data[0])

    # Step 1: Processing Data
    # X_text is the questions for each training data
    X_text = [preprocess_text(item["question"]) for item in training_data]
    # Y is the binary vector representation of the topics for each question (2D Numpy Array)
    Y = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

    # Step 2: Encode questions using SentenceTransformer
    X_embedding = model.encode(X_text)

    # Step 3: Scale the encoded embeddings
    scaler = StandardScaler()   # Standardize features by removing the mean and scaling to unit variance
    X_scaled = scaler.fit_transform(X_embedding)

    # Step 4: Dimensionality Reduction using PCA
    pca = PCA(n_components=0.95)  # Retain 95% of variance Aim: to reduce the number of features while retaining the variance
    X_reduced = pca.fit_transform(X_scaled)

    # Step 3: Split the data into training and test sets
    # X_test and X train is the embedding of each question, where the first dimension is the question and the second dimension is the embedding
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=41)

    #DEBUG: debugging hash map
    hash_map = {}
    for i in range(len(y_train)):
        for j in range(len(y_train[0])):
            if j not in hash_map:
                hash_map[j] = 0
            elif y_train[i][j] == 1:
                hash_map[j] += 1
    print(hash_map)
    # Step 4: Train a Multi-Label Classifier

    # #DEBUG: For debugging
    # np.set_printoptions(threshold=np.inf)
    # print(y_train)

    # NOTE: You need to have the sufficient data and you need to have at least one row for each topics in all_topics for y_train

    # base_classifier = NearestCentroid()
    base_classifier = LogisticRegression(max_iter=500)

    # base_classifier = RandomForestClassifier(n_estimators=270, random_state=42)
    multi_label_classifier = MultiOutputClassifier(base_classifier)
    multi_label_classifier.fit(X_train, y_train)   # trains the multi_label_classifier

    # Save the trained model
    joblib.dump(multi_label_classifier, 'multi_label_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')

    # Step 5: Predict for a New Question
    # Intepretation of predicted_probabilities
    # This array corresponds to the predicted probabilities for one of the labels (topics).
    # Each row in the array corresponds to a sample in the test set.
    # The first column contains the probabilities for class 0 (the negative class).
    # The second column contains the probabilities for class 1 (the positive class).
    predicted_probabilities = multi_label_classifier.predict_proba(X_test)

    # Step 6: Assign Topics Based on Threshold
    threshold = 0.1  # Adjust this threshold as needed #TODO: need a better way to determine threshold
    # for each pp, for each of it's probability, check the index 0 against threshold
    predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]

    # Currently, predicted labels is a list of numpy arrays, where each array contains the predicted labels for each topic. e.g. [array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0])]. Each array is for the topics, while the 1 and 0 indicates FOR each questions FOR THIS TOPIC
    # We transposed it to make it easier to get the topics for each question
    # Now each inner array is for one question, and each 1 and 0 is for each topics for that QUESTION
    transposed_labels = np.array(predicted_labels).T

    # Step 7: Get topic names for each question
    question_predictions = []
    for q in transposed_labels:
        predicted_topics = [all_topics[i] for i, label in enumerate(q) if label == 1]
        question_predictions.append(predicted_topics)

    f1_scores = f1_score(y_test, transposed_labels, average='micro')
    recall_scores = recall_score(y_test, transposed_labels, average='micro')
    precision_scores = precision_score(y_test, transposed_labels, average='micro')

    #TODO: mention the steps taken to improve in r
    print("Training Topics Prediction Model Complete and Saved")
    print(f"F1-Score for Multi-Label Topic Classification: {f1_scores}")
    print(f"Recall-Score for Multi-Label Topic Classification: {recall_scores}")
    print(f"Precision-Score for Multi-Label Topic Classification: {precision_scores}")


# Function to predict topics for new questions
# TODO: Good to abstract into pipeline?
def predict_topics(new_questions):
    # Load the trained model
    multi_label_classifier = joblib.load('multi_label_classifier.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    
    # Encode new questions using the same SentenceTransformer model
    new_embeddings = model.encode(new_questions)

    # Scale the encoded embeddings using the already fitted scaler
    scaled_embedding = scaler.transform(new_embeddings)

    # Reduce dimensionality using the already fitted PCA
    reduced_embedding = pca.transform(scaled_embedding)
    
    # Predict probabilities
    predicted_probabilities = multi_label_classifier.predict_proba(reduced_embedding)
    
    # Assign topics based on threshold
    threshold = 0.15
    predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]
    transposed_labels = np.array(predicted_labels).T
    
    # Get topic names for each question
    question_predictions = []
    for q in transposed_labels:
        predicted_topics = [all_topics[i] for i, label in enumerate(q) if label == 1]
        question_predictions.append(predicted_topics)
    
    return question_predictions


# # NOTE: Lazy Predict Code Runner
# # NOTE: Warning, this will not work without the lazy predict library (not included in requirements.txt)
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split
# from topics_data import all_topics
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# '''
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
# '''

# ### Helper functions
# # Return an array of binary values indicating the presence of each topic in the list of all topics
# def topics_to_vector(topics, all_topics):
#     return [1 if topic.strip() in topics else 0 for topic in all_topics]

# def load_training_data(csv_file_path):
#     df = pd.read_excel(csv_file_path, engine='openpyxl')
    
#     def split_topics(topics):
#         return topics.split("; ")

#     # df["topics"] = df["topics"].apply(split_topics)   # Split topics by ;
#     df = df.to_dict(orient='records')    # Convert DataFrame to a list of dictionaries
#     return df



# training_data = load_training_data('training_data.xlsx')

# # Step 1: Processing Data
# # X_text is the questions for each training data
# X_text = [item["question"] for item in training_data]
# # Y is the binary vector representation of the topics for each question (2D Numpy Array)
# Y = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

# # Step 2: Encode questions using SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# X_embedding = model.encode(X_text)

# # Step 3: Scale the encoded embeddings
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_embedding)

# # Step 4: Dimensionality Reduction using PCA
# pca = PCA(n_components=0.95)  # Retain 95% of variance
# X_reduced = pca.fit_transform(X_scaled)


# print(Y)
# Y = [item["topics"] for item in training_data]

# # Step 3: Split the data into training and test sets
# # X_test and X train is the embedding of each question, where the first dimension is the question and the second dimension is the embedding
# X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=42)

# #DEBUG: debugging hash map
# # hash_map = {}
# # for i in range(len(y_train)):
# #     for j in range(len(y_train[0])):
# #         if j not in hash_map:
# #             hash_map[j] = 0
# #         elif y_train[i][j] == 1:
# #             hash_map[j] += 1
# # print(hash_map)
# # Step 4: Train a Multi-Label Classifier

# # #DEBUG: For debugging
# # np.set_printoptions(threshold=np.inf)
# # print(y_train)

# # NOTE: You need to have the sufficient data and you need to have at least one row for each topics in all_topics for y_train

# from lazypredict.Supervised import LazyClassifier
# clf = LazyClassifier(verbose=0,ignore_warnings=False, custom_metric=None)
# models,predictions = clf.fit(X_train, X_test, y_train, y_test)

# print(models)

# # # base_classifier = LogisticRegression(max_iter=1000)
# # base_classifier = RandomForestClassifier(n_estimators=270, random_state=42)
# # multi_label_classifier = MultiOutputClassifier(base_classifier)
# # multi_label_classifier.fit(X_train, y_train)   # trains the multi_label_classifier

# # # Save the trained model
# # joblib.dump(multi_label_classifier, 'multi_label_classifier.pkl')

# # # Step 5: Predict for a New Question
# # # Intepretation of predicted_probabilities
# # # This array corresponds to the predicted probabilities for one of the labels (topics).
# # # Each row in the array corresponds to a sample in the test set.
# # # The first column contains the probabilities for class 0 (the negative class).
# # # The second column contains the probabilities for class 1 (the positive class).
# # predicted_probabilities = multi_label_classifier.predict_proba(X_test)

# # # Step 6: Assign Topics Based on Threshold
# # threshold = 0.15  # Adjust this threshold as needed
# # # for each pp, for each of it's probability, check the index 0 against threshold
# # predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]

# # # Currently, predicted labels is a list of numpy arrays, where each array contains the predicted labels for each topic. e.g. [array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0])]. Each array is for the topics, while the 1 and 0 indicates FOR each questions FOR THIS TOPIC
# # # We transposed it to make it easier to get the topics for each question
# # # Now each inner array is for one question, and each 1 and 0 is for each topics for that QUESTION
# # transposed_labels = np.array(predicted_labels).T

# # # Step 7: Get topic names for each question
# # question_predictions = []
# # for q in transposed_labels:
# #     predicted_topics = [all_topics[i] for i, label in enumerate(q) if label == 1]
# #     question_predictions.append(predicted_topics)

# # f1_scores = f1_score(y_test, transposed_labels, average='micro')

# # print("Training Topics Prediction Model Complete and Saved")
# # print(f"F1-Score for Multi-Label Topic Classification: {f1_scores}")


# # # Function to predict topics for new questions
# # # TODO: Good to abstract into pipeline?
# # def predict_topics(new_questions):
# #     # Load the trained model
# #     multi_label_classifier = joblib.load('multi_label_classifier.pkl')
    
# #     # Encode new questions using the same SentenceTransformer model
# #     new_embeddings = model.encode(new_questions)
    
# #     # Predict probabilities
# #     predicted_probabilities = multi_label_classifier.predict_proba(new_embeddings)
    
# #     # Assign topics based on threshold
# #     threshold = 0.15
# #     predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]
# #     transposed_labels = np.array(predicted_labels).T
    
# #     # Get topic names for each question
# #     question_predictions = []
# #     for q in transposed_labels:
# #         predicted_topics = [all_topics[i] for i, label in enumerate(q) if label == 1]
# #         question_predictions.append(predicted_topics)
    
# #     return question_predictions

