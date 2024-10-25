import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify import apply_features
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
import pickle

def extract_features(text):
    words = word_tokenize(text)
    return {word: True for word in words}

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        symptoms, disease = line.strip().split('\t')
        features = extract_features(symptoms)
        data.append((features, disease))
    
    return data

def train_classifier(train_data):
    return NaiveBayesClassifier.train(train_data)

if __name__ == "__main__":
    train_data = load_data(r'C:\Users\preeti\OneDrive\Desktop\medical_chatbot\data\processed_data.txt')
    classifier = train_classifier(train_data)
    # Save the trained model
    with open(r'C:\Users\preeti\OneDrive\Desktop\medical_chatbot\data\medical_chatbot_model', 'wb') as f:
        pickle.dump(classifier, f)
