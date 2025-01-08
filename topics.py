# import numpy as np
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
# from sentence_transformers import SentenceTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score

# # Using Pre-Trained Sentence Transformer:

# # The transformer is already trained on large datasets to produce meaningful text embeddings.
# # You’re leveraging this capability without modifying the model's parameters.
# # Training a Separate Classifier:

# # After generating embeddings, you're training a downstream classifier (e.g., a MultiOutputClassifier) to map embeddings to the labels (topics in your case).
# # Only the classifier learns from your labeled data.

# # The Sentence Transformer is Pretrained:

# # It has already been trained on a large dataset (e.g., natural language inference, paraphrase detection) to generate high-quality embeddings that capture the semantic meaning of text.
# # You are using this pre-trained model "as is" to convert text into dense vector embeddings. These embeddings are fixed and are not updated during the downstream classifier training.
# # The Downstream Classifier is Not Pretrained:

# # The downstream classifier (e.g., Logistic Regression, Random Forest, MultiOutputClassifier, etc.) starts with random or default initial parameters.
# # It is trained (using .fit) on your task-specific dataset, where it learns to map the embeddings to the desired output labels (e.g., topics in your case).
# # This training only updates the parameters of the downstream classifier, not the Sentence Transformer.

# # Step 1: Training Data
# training_data = [
#     {"question": "What is the capital of France?", "topics": ["Geography", "History"]},
#     {"question": "How does photosynthesis work?", "topics": ["Biology"]},
#     {"question": "Explain the process of osmosis.", "topics": ["Biology", "Chemistry"]},
#     # {"question": "What are the benefits of exercise?", "topics": ["Health"]},
#     # {"question": "Who discovered penicillin?", "topics": ["Biology", "History"]},
#     # {"question": "What are the continents of the world?", "topics": ["Geography"]},
#     # {"question": "Describe the water cycle.", "topics": ["Geography", "Biology"]},
#     # {"question": "What is the function of the human heart?", "topics": ["Biology", "Health"]},
#     # {"question": "Which elements are in the periodic table?", "topics": ["Chemistry"]},
#     # {"question": "How does a chemical reaction occur?", "topics": ["Chemistry", "Biology"]},
#     # {"question": "What are the effects of smoking on the lungs?", "topics": ["Health", "Biology"]},
#     # {"question": "What are the main causes of World War I?", "topics": ["History"]},
#     # {"question": "What are the major rivers in Africa?", "topics": ["Geography"]},
#     # {"question": "Explain the importance of a balanced diet.", "topics": ["Health"]},
#     # {"question": "What are the effects of climate change?", "topics": ["Geography", "Biology"]},
#     # {"question": "Who was involved in the American Revolution?", "topics": ["History"]},
#     # {"question": "How do vaccines work?", "topics": ["Biology", "Health"]},
#     # {"question": "Explain Newton's laws of motion.", "topics": ["Physics", "History"]},
#     # {"question": "What are the uses of helium gas?", "topics": ["Chemistry"]},
#     # {"question": "Describe the Great Wall of China.", "topics": ["History", "Geography"]},
#     # {"question": "How do plants absorb nutrients from soil?", "topics": ["Biology", "Chemistry"]},
#     # {"question": "What is the structure of the human brain?", "topics": ["Biology", "Health"]},
#     # {"question": "What are the causes of acid rain?", "topics": ["Geography", "Chemistry"]},
#     # {"question": "Who painted the Mona Lisa?", "topics": ["History", "Art"]},
#     # {"question": "What are the properties of water?", "topics": ["Chemistry"]},
#     # {"question": "What is the significance of the Nile River?", "topics": ["Geography", "History"]},
#     # {"question": "What are the benefits of regular physical activity?", "topics": ["Health"]},
#     # {"question": "Describe the process of evolution.", "topics": ["Biology", "History"]},
#     # {"question": "What is the importance of vitamins in our diet?", "topics": ["Health", "Biology"]},
#     # {"question": "What is the role of mitochondria in cells?", "topics": ["Biology"]},
#     # {"question": "Explain the history of the Industrial Revolution.", "topics": ["History"]},
#     # {"question": "How do volcanoes form?", "topics": ["Geography", "Geology"]},
#     # {"question": "What is the greenhouse effect?", "topics": ["Geography", "Biology"]},
#     # {"question": "What is the theory of relativity?", "topics": ["Physics", "History"]},
#     # {"question": "Describe the structure of an atom.", "topics": ["Chemistry", "Physics"]},
#     # {"question": "What are the dangers of drug abuse?", "topics": ["Health", "Biology"]},
#     # {"question": "Who discovered gravity?", "topics": ["History", "Physics"]},
#     # {"question": "What are the layers of the Earth?", "topics": ["Geography", "Geology"]},
#     # {"question": "How does pollution affect marine life?", "topics": ["Biology", "Geography"]},
#     # {"question": "What are the principles of quantum mechanics?", "topics": ["Physics"]},
# ]

# all_topics = [
#     "Geography", "History", "Biology", "Chemistry"
# ]

# # all_topics = [
# #     "Geography", "History", "Biology", "Chemistry", 
# #     "Health", "Physics", "Art", "Geology"
# # ]

# # Step 2: Convert topics to binary vectors
# def topics_to_vector(topics, all_topics):
#     return [1 if topic in topics else 0 for topic in all_topics]

# X_text = [item["question"] for item in training_data]
# topic_labels = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

# print("printing topic labels")
# print(topic_labels)


# # Step 3: Encode questions using SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# question_embeddings = model.encode(X_text)

# # Step 2: Split the data into training and test sets
# # X_train, X_test, y_train_topics, y_test_topics = train_test_split(question_embeddings, topic_labels, test_size=0.2, random_state=42)

# # NOTE: test for accuracy (to remove)
# X_train = question_embeddings
# y_train_topics = topic_labels

# print("printing length")
# print(len(topic_labels))
# print(len(topic_labels[0]))

# X_test = question_embeddings
# y_test_topics = topic_labels

# # Step 3: Train multi-label classifier for topics
# base_classifier = LogisticRegression(max_iter=1000)
# topic_classifier = MultiOutputClassifier(base_classifier)
# topic_classifier.fit(X_train, y_train_topics)


# # Step 5: Predict for a New Question
# # new_question = "How is volcanic eruption formed?"
# # new_question = "What is the capital of France?"
# # new_question = "How does photosynthesis work?"
# # new_question = "What is the function of the human heart?"
# # new_question = "What are the effects of smoking on the lungs?"

# #QUESTION: why y_train_topics has 4 columns/topics for each question/row but predicted_probability only has 3 columns/topics for each question/row
# # new_question_embedding = model.encode([new_question])
# predicted_probabilities = topic_classifier.predict_proba(X_test)

# print("Predicted Probabilities:", predicted_probabilities)

# # Step 6: Assign Topics Based on Threshold
# threshold = 0.4  # Adjust this threshold as needed
# predicted_labels = [(pp >= threshold).astype(int) for pp in predicted_probabilities]

# print(predicted_labels)

# # Flatten predicted_labels to ensure it's 1D
# # predicted_labels = predicted_labels.flatten()


# question_prediction = []
# for q in predicted_labels:
#     print("printing q", q)
#     predicted_topics = [all_topics[i] for i, label in enumerate(q) if label.flatten()[1] == 1]
#     question_prediction.append(predicted_topics)


# # Get topic names
# # predicted_topics = [all_topics[i] for i, label in enumerate(predicted_labels) if label.flatten()[1] == 1]
# print("Predicted Topics:", question_prediction)

# # Step 5: Evaluate model performance using F1 Score (multi-label)
# print("printing f1 scores")
# print(y_test_topics)
# print(predicted_labels)
# to_verify_y_labels = []
# for q in predicted_labels:
#     # print("printing q")
#     # print(len(q))
#     # print(q)

#     to_verify_y_label = [0 if tuple[1] == 0 else 1 for tuple in q]
#     to_verify_y_labels.append(to_verify_y_label)

# print("printing to verify y labels")
# print(y_test_topics)
# print(to_verify_y_labels)


# f1_scores = f1_score(y_test_topics, to_verify_y_labels, average='micro')
# print(f"F1-Score for Multi-Label Topic Classification: {f1_scores}")





# # NOTE: OG CODE


# import numpy as np
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics import f1_score


# # Using Pre-Trained Sentence Transformer:

# # The transformer is already trained on large datasets to produce meaningful text embeddings.
# # You’re leveraging this capability without modifying the model's parameters.
# # Training a Separate Classifier:

# # After generating embeddings, you're training a downstream classifier (e.g., a MultiOutputClassifier) to map embeddings to the labels (topics in your case).
# # Only the classifier learns from your labeled data.

# # The Sentence Transformer is Pretrained:

# # It has already been trained on a large dataset (e.g., natural language inference, paraphrase detection) to generate high-quality embeddings that capture the semantic meaning of text.
# # You are using this pre-trained model "as is" to convert text into dense vector embeddings. These embeddings are fixed and are not updated during the downstream classifier training.
# # The Downstream Classifier is Not Pretrained:

# # The downstream classifier (e.g., Logistic Regression, Random Forest, MultiOutputClassifier, etc.) starts with random or default initial parameters.
# # It is trained (using .fit) on your task-specific dataset, where it learns to map the embeddings to the desired output labels (e.g., topics in your case).
# # This training only updates the parameters of the downstream classifier, not the Sentence Transformer.

# # Step 1: Training Data
# training_data = [
#     {"question": "What is the capital of France?", "topics": ["Geography", "History"]},
#     {"question": "How does photosynthesis work?", "topics": ["Biology"]},
#     {"question": "Explain the process of osmosis.", "topics": ["Biology", "Chemistry"]},
#     {"question": "What are the benefits of exercise?", "topics": ["Health"]},
#     {"question": "Who discovered penicillin?", "topics": ["Biology", "History"]},
#     {"question": "What are the continents of the world?", "topics": ["Geography"]},
#     {"question": "Describe the water cycle.", "topics": ["Geography", "Biology"]},
#     {"question": "What is the function of the human heart?", "topics": ["Biology", "Health"]},
#     {"question": "Which elements are in the periodic table?", "topics": ["Chemistry"]},
#     {"question": "How does a chemical reaction occur?", "topics": ["Chemistry", "Biology"]},
#     {"question": "What are the effects of smoking on the lungs?", "topics": ["Health", "Biology"]},
#     {"question": "What are the main causes of World War I?", "topics": ["History"]},
#     {"question": "What are the major rivers in Africa?", "topics": ["Geography"]},
#     {"question": "Explain the importance of a balanced diet.", "topics": ["Health"]},
#     {"question": "What are the effects of climate change?", "topics": ["Geography", "Biology"]},
#     {"question": "Who was involved in the American Revolution?", "topics": ["History"]},
#     {"question": "How do vaccines work?", "topics": ["Biology", "Health"]},
#     {"question": "Explain Newton's laws of motion.", "topics": ["Physics", "History"]},
#     {"question": "What are the uses of helium gas?", "topics": ["Chemistry"]},
#     {"question": "Describe the Great Wall of China.", "topics": ["History", "Geography"]},
#     {"question": "How do plants absorb nutrients from soil?", "topics": ["Biology", "Chemistry"]},
#     {"question": "What is the structure of the human brain?", "topics": ["Biology", "Health"]},
#     {"question": "What are the causes of acid rain?", "topics": ["Geography", "Chemistry"]},
#     {"question": "Who painted the Mona Lisa?", "topics": ["History", "Art"]},
#     {"question": "What are the properties of water?", "topics": ["Chemistry"]},
#     {"question": "What is the significance of the Nile River?", "topics": ["Geography", "History"]},
#     {"question": "What are the benefits of regular physical activity?", "topics": ["Health"]},
#     {"question": "Describe the process of evolution.", "topics": ["Biology", "History"]},
#     {"question": "What is the importance of vitamins in our diet?", "topics": ["Health", "Biology"]},
#     {"question": "What is the role of mitochondria in cells?", "topics": ["Biology"]},
#     {"question": "Explain the history of the Industrial Revolution.", "topics": ["History"]},
#     {"question": "How do volcanoes form?", "topics": ["Geography", "Geology"]},
#     {"question": "What is the greenhouse effect?", "topics": ["Geography", "Biology"]},
#     {"question": "What is the theory of relativity?", "topics": ["Physics", "History"]},
#     {"question": "Describe the structure of an atom.", "topics": ["Chemistry", "Physics"]},
#     {"question": "What are the dangers of drug abuse?", "topics": ["Health", "Biology"]},
#     {"question": "Who discovered gravity?", "topics": ["History", "Physics"]},
#     {"question": "What are the layers of the Earth?", "topics": ["Geography", "Geology"]},
#     {"question": "How does pollution affect marine life?", "topics": ["Biology", "Geography"]},
#     {"question": "What are the principles of quantum mechanics?", "topics": ["Physics"]},
# ]

# all_topics = [
#     "Geography", "History", "Biology", "Chemistry", 
#     "Health", "Physics", "Art", "Geology"
# ]

# #TODO: to handle multiple questions, + proper test split and training split

# # Step 2: Convert topics to binary vectors
# def topics_to_vector(topics, all_topics):
#     return [1 if topic in topics else 0 for topic in all_topics]

# X_text = [item["question"] for item in training_data]
# Y = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

# print(Y)

# # Step 3: Encode questions using SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# X_embeddings = model.encode(X_text)

# # Step 4: Train a Multi-Label Classifier
# base_classifier = LogisticRegression(max_iter=1000)
# multi_label_classifier = MultiOutputClassifier(base_classifier)
# multi_label_classifier.fit(X_embeddings, Y)   # trains the classifier

# # Step 5: Predict for a New Question
# # new_question = "What is the capital of France?"
# # new_question = "What is the function of the human heart?"
# # new_question = "Explain the process of osmosis."
# new_question = "Explain Newton's laws of motion."

# new_question_embedding = model.encode([new_question])
# predicted_probabilities = multi_label_classifier.predict_proba(new_question_embedding)

# # print("Predicted Probabilities:", predicted_probabilities)




# # Step 6: Assign Topics Based on Threshold
# threshold = 0.4  # Adjust this threshold as needed
# predicted_labels = [(pp >= threshold).astype(int) for pp in predicted_probabilities]

# # print(predicted_labels)

# # Flatten predicted_labels to ensure it's 1D
# # predicted_labels = predicted_labels.flatten()

# # Get topic names
# predicted_topics = [all_topics[i] for i, label in enumerate(predicted_labels) if label.flatten()[1] == 1]
# print("Predicted Topics:", predicted_topics)

# to_verify_label = [1 if label.flatten()[1] == 1 else 0 for label in predicted_labels]
# print(to_verify_label)

# f1_scores = f1_score([1, 1, 0, 0, 0, 0, 0, 0], to_verify_label, average='micro')
# print(f"F1-Score for Multi-Label Topic Classification: {f1_scores}")


# # NOTE: 8/1/2025


# import numpy as np
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split


# # Using Pre-Trained Sentence Transformer:

# # The transformer is already trained on large datasets to produce meaningful text embeddings.
# # You’re leveraging this capability without modifying the model's parameters.
# # Training a Separate Classifier:

# # After generating embeddings, you're training a downstream classifier (e.g., a MultiOutputClassifier) to map embeddings to the labels (topics in your case).
# # Only the classifier learns from your labeled data.

# # The Sentence Transformer is Pretrained:

# # It has already been trained on a large dataset (e.g., natural language inference, paraphrase detection) to generate high-quality embeddings that capture the semantic meaning of text.
# # You are using this pre-trained model "as is" to convert text into dense vector embeddings. These embeddings are fixed and are not updated during the downstream classifier training.
# # The Downstream Classifier is Not Pretrained:

# # The downstream classifier (e.g., Logistic Regression, Random Forest, MultiOutputClassifier, etc.) starts with random or default initial parameters.
# # It is trained (using .fit) on your task-specific dataset, where it learns to map the embeddings to the desired output labels (e.g., topics in your case).
# # This training only updates the parameters of the downstream classifier, not the Sentence Transformer.

# # Step 1: Training Data
# training_data = [
#     # Geography
#     {"question": "What factors contribute to the formation of river deltas?", "topics": ["Geography"]},
#     {"question": "How do urban heat islands impact local climates?", "topics": ["Geography", "Environmental Science"]},
#     {"question": "What are the main causes of coastal erosion?", "topics": ["Geography", "Environmental Science"]},
#     {"question": "How does plate tectonics influence the formation of mountains?", "topics": ["Geography", "Geology"]},
#     {"question": "What is the significance of latitude and longitude in navigation?", "topics": ["Geography"]},
#     {"question": "What is the role of monsoons in agricultural practices?", "topics": ["Geography"]},
#     {"question": "How do glaciers shape the Earth's surface?", "topics": ["Geography", "Geology"]},
#     {"question": "What are the environmental impacts of mining on landscapes?", "topics": ["Geography", "Environmental Science"]},
#     {"question": "What is the importance of water resources in arid regions?", "topics": ["Geography"]},
#     {"question": "How are natural disasters like tsunamis and earthquakes monitored?", "topics": ["Geography", "Technology"]},

#     # History
#     {"question": "What were the economic consequences of the Industrial Revolution?", "topics": ["History", "Economics"]},
#     {"question": "What were the causes and effects of the Cold War?", "topics": ["History"]},
#     {"question": "How did the Renaissance influence European culture?", "topics": ["History", "Art"]},
#     {"question": "What were the key achievements of ancient Egyptian civilization?", "topics": ["History"]},
#     {"question": "What were the social impacts of the abolition of slavery?", "topics": ["History", "Sociology"]},
#     {"question": "What role did the printing press play in the spread of knowledge?", "topics": ["History", "Technology"]},
#     {"question": "How did colonization impact indigenous populations?", "topics": ["History", "Sociology"]},
#     {"question": "What were the key events leading to the fall of the Roman Empire?", "topics": ["History"]},
#     {"question": "How did the Silk Road contribute to cultural exchange?", "topics": ["History", "Geography"]},
#     {"question": "What was the significance of the American Civil Rights Movement?", "topics": ["History"]},

#     # Biology
#     {"question": "What are the different types of cell division and their purposes?", "topics": ["Biology"]},
#     {"question": "How does photosynthesis work at the molecular level?", "topics": ["Biology", "Chemistry"]},
#     {"question": "What are the ecological roles of keystone species?", "topics": ["Biology", "Environmental Science"]},
#     {"question": "How do hormones regulate bodily functions?", "topics": ["Biology", "Health"]},
#     {"question": "What are the impacts of habitat fragmentation on wildlife?", "topics": ["Biology", "Environmental Science"]},
#     {"question": "How does genetic variation arise within populations?", "topics": ["Biology"]},
#     {"question": "What are the stages of mitosis and their significance?", "topics": ["Biology"]},
#     {"question": "How do plants adapt to extreme environments?", "topics": ["Biology", "Geography"]},
#     {"question": "How do immune cells recognize pathogens?", "topics": ["Biology", "Health"]},
#     {"question": "What are the main steps in protein synthesis?", "topics": ["Biology"]},

#     # Chemistry
#     {"question": "What are the key principles of chemical equilibrium?", "topics": ["Chemistry"]},
#     {"question": "How do catalysts speed up chemical reactions?", "topics": ["Chemistry"]},
#     {"question": "What are the main components of an electrochemical cell?", "topics": ["Chemistry"]},
#     {"question": "How do acids and bases interact in neutralization reactions?", "topics": ["Chemistry"]},
#     {"question": "What is the environmental impact of ozone-depleting chemicals?", "topics": ["Chemistry", "Environmental Science"]},
#     {"question": "What are the differences between ionic and covalent bonds?", "topics": ["Chemistry"]},
#     {"question": "How do solubility rules predict the formation of precipitates?", "topics": ["Chemistry"]},
#     {"question": "What are the industrial applications of polymers?", "topics": ["Chemistry", "Technology"]},
#     {"question": "What are the principles of organic reaction mechanisms?", "topics": ["Chemistry"]},
#     {"question": "How does the periodic table organize elements?", "topics": ["Chemistry"]},

#     # Health
#     {"question": "What are the benefits of regular physical activity?", "topics": ["Health"]},
#     {"question": "How does nutrition affect overall health?", "topics": ["Health"]},
#     {"question": "What are the causes and prevention methods for diabetes?", "topics": ["Health"]},
#     {"question": "How do vaccines help prevent diseases?", "topics": ["Health", "Biology"]},
#     {"question": "What are the mental health effects of chronic stress?", "topics": ["Health", "Psychology"]},
#     {"question": "What are the risks associated with smoking and alcohol consumption?", "topics": ["Health", "Biology"]},
#     {"question": "How does sleep affect cognitive function?", "topics": ["Health", "Psychology"]},
#     {"question": "What is the importance of hydration for physical performance?", "topics": ["Health"]},
#     {"question": "How do wearable devices help track fitness?", "topics": ["Health", "Technology"]},
#     {"question": "What are the early symptoms of cardiovascular diseases?", "topics": ["Health"]},

#     # Physics
#     {"question": "What are the key principles of Newton's laws of motion?", "topics": ["Physics"]},
#     {"question": "How does energy transfer occur in different forms?", "topics": ["Physics"]},
#     {"question": "What is the Doppler effect, and where is it observed?", "topics": ["Physics"]},
#     {"question": "How do gravitational waves provide information about the universe?", "topics": ["Physics", "Astronomy"]},
#     {"question": "What are the key differences between classical and quantum physics?", "topics": ["Physics"]},
#     {"question": "How do lasers work, and what are their applications?", "topics": ["Physics", "Technology"]},
#     {"question": "What is the concept of wave-particle duality?", "topics": ["Physics"]},
#     {"question": "How do superconductors function at low temperatures?", "topics": ["Physics", "Chemistry"]},
#     {"question": "What are the principles of thermodynamics?", "topics": ["Physics"]},
#     {"question": "How does the curvature of space-time affect gravity?", "topics": ["Physics", "Astronomy"]},

#     # Art
#     {"question": "What are the key characteristics of Renaissance art?", "topics": ["Art"]},
#     {"question": "How does abstract art differ from realism?", "topics": ["Art"]},
#     {"question": "What are the cultural influences in traditional Chinese paintings?", "topics": ["Art", "History"]},
#     {"question": "How does digital art impact modern design trends?", "topics": ["Art", "Technology"]},
#     {"question": "What are the techniques used in oil painting?", "topics": ["Art"]},
#     {"question": "What is the significance of color theory in visual art?", "topics": ["Art"]},
#     {"question": "What is the role of public art in urban spaces?", "topics": ["Art", "Sociology"]},
#     {"question": "How does art influence political movements?", "topics": ["Art", "History"]},
#     {"question": "What is the symbolism in Van Gogh’s Starry Night?", "topics": ["Art"]},
#     {"question": "What are the major contributions of modern sculpture?", "topics": ["Art"]}
# ]

# all_topics = [
#     "Geography", "History", "Biology", "Chemistry", 
#     "Health", "Physics", "Art"
# ]


# #TODO: to handle multiple questions, + proper test split and training split

# # Step 2: Convert topics to binary vectors
# def topics_to_vector(topics, all_topics):
#     return [1 if topic in topics else 0 for topic in all_topics]

# X_text = [item["question"] for item in training_data]
# Y = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

# print(Y)

# # Step 3: Encode questions using SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# X_embeddings = model.encode(X_text)

# #NOTE: testing
# # Step 4: Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_embeddings, Y, test_size=0.2, random_state=42)

# print("printing y_test")
# print(y_test)

# # Step 4: Train a Multi-Label Classifier
# base_classifier = LogisticRegression(max_iter=3000)
# multi_label_classifier = MultiOutputClassifier(base_classifier)
# multi_label_classifier.fit(X_embeddings, Y)   # trains the classifier

# # Step 5: Predict for a New Question
# # new_question = "What is the capital of France?"
# # new_question = "What is the function of the human heart?"
# # new_question = "Explain the process of osmosis."
# # new_question = "Explain Newton's laws of motion."
# new_question = "What are the key characteristics of Renaissance art?"
# new_question_embedding = model.encode([new_question])

# #NOTE: testing
# new_questions_embedding = model.encode(X_text)

# predicted_probabilities = multi_label_classifier.predict_proba(new_questions_embedding)
# print("printing predicted probabilities")
# print(predicted_probabilities)


# # Step 6: Assign Topics Based on Threshold
# threshold = 0.4  # Adjust this threshold as needed
# predicted_labels = [(pp >= threshold).astype(int) for pp in predicted_probabilities]

# # Get topic names
# predicted_topics = [all_topics[i] for i, label in enumerate(predicted_labels) if label.flatten()[1] == 1]
# print("Predicted Topics:", predicted_topics)

# to_verify_label = [1 if label.flatten()[1] == 1 else 0 for label in predicted_labels]
# print(to_verify_label)

# # f1_scores = f1_score(y_test, to_verify_label, average='micro')
# # print(f"F1-Score for Multi-Label Topic Classification: {f1_scores}")

# NOTE: 8/1/2025 With MultiQuestion Prediction and Proper Training/Test Split
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


# Step 1: Training Data



# Step 2: Convert topics to binary vectors
def topics_to_vector(topics, all_topics):
    return [1 if topic in topics else 0 for topic in all_topics]

X_text = [item["question"] for item in training_data]
Y = np.array([topics_to_vector(item["topics"], all_topics) for item in training_data])

# print(Y)

# Step 3: Encode questions using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = model.encode(X_text)

#NOTE: testing
# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, Y, test_size=0.2, random_state=42)

print("printing y_test")
print(y_test)

print("number of questions in y_test")
print(len(y_test))

# Step 4: Train a Multi-Label Classifier
base_classifier = LogisticRegression(max_iter=3000)
multi_label_classifier = MultiOutputClassifier(base_classifier)
multi_label_classifier.fit(X_train, y_train)   # trains the classifier

# Step 5: Predict for a New Question
# new_question = "What is the capital of France?"
# new_question = "What is the function of the human heart?"
# new_question = "Explain the process of osmosis."
# new_question = "Explain Newton's laws of motion."
# new_question = "What are the key characteristics of Renaissance art?"
# new_question_embedding = model.encode([new_question])

print("printing x_test")
print(X_test)
print("x_test length")
print(len(X_test))



#NOTE: testing
# new_questions_embedding = model.encode(X_test)


predicted_probabilities = multi_label_classifier.predict_proba(X_test)
print("printing predicted probabilities")
print(predicted_probabilities)


# Step 6: Assign Topics Based on Threshold
threshold = 0.2  # Adjust this threshold as needed
# predicted_labels = [(pp >= threshold).astype(int) for pp in predicted_probabilities]
predicted_labels = [(pp[:, 1] >= threshold).astype(int) for pp in predicted_probabilities]

print("printing predicted labels")
print(predicted_labels)

transposed_labels = np.array(predicted_labels).T
print("printing transposed")
print(transposed_labels)

# Get topic names for each question
question_predictions = []
to_verify_labels = []
for q in transposed_labels:
    predicted_topics = [all_topics[i] for i, label in enumerate(q) if label == 1]
    question_predictions.append(predicted_topics)

print("Predicted Topics:", question_predictions)


f1_scores = f1_score(y_test, transposed_labels, average='micro')
print(f"F1-Score for Multi-Label Topic Classification: {f1_scores}")

