"""
This script is used to call the training of the topics identification model. 
You can train the model by running python train_model.py in bash. 
"""

from topics import train_classifier

if __name__ == "__main__":
    print("Training Topic Identification Model Script Called")
    train_classifier()