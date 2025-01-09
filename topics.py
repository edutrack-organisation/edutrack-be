import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from topics_data import training_data, all_topics

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
    return [1 if topic in topics else 0 for topic in all_topics]

# Step 1: Processing Data
# X_text is the questions for each training data
X_text = [item["question"] for item in training_data]
# Y is the binary vector representation of the topics for each question (2D Numpy Array)
Y = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

# Step 2: Encode questions using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
X_embedding = model.encode(X_text)

# Step 3: Split the data into training and test sets
# X_test and X train is the embedding of each question, where the first dimension is the question and the second dimension is the embedding
X_train, X_test, y_train, y_test = train_test_split(X_embedding, Y, test_size=0.2, random_state=42)

# Step 4: Train a Multi-Label Classifier
base_classifier = LogisticRegression(max_iter=1000)
multi_label_classifier = MultiOutputClassifier(base_classifier)
multi_label_classifier.fit(X_train, y_train)   # trains the multi_label_classifier

# Step 5: Predict for a New Question
# Intepretation of predicted_probabilities
# This array corresponds to the predicted probabilities for one of the labels (topics).
# Each row in the array corresponds to a sample in the test set.
# The first column contains the probabilities for class 0 (the negative class).
# The second column contains the probabilities for class 1 (the positive class).
predicted_probabilities = multi_label_classifier.predict_proba(X_test)

# Step 6: Assign Topics Based on Threshold
threshold = 0.15  # Adjust this threshold as needed
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

print("Predicted Topics:", question_predictions)
print(f"F1-Score for Multi-Label Topic Classification: {f1_scores}")

