import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class SemanticAnalysisDemonstrator:
    def __init__(self):
        self.model = None
        self.label_encoder = None
    
    def preprocess_text(self, text):
        """Simple text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove basic stopwords
        stop_words = set(['the', 'a', 'an', 'in', 'to', 'for'])
        words = text.split()
        words = [word for word in words if word not in stop_words]
        
        return ' '.join(words)
    
    def load_dataset(self):
        """Load dataset from data.csv"""
        df = pd.read_csv('data.csv', encoding='latin1')
        # Keep only necessary columns
        df = df[['text', 'sentiment']]
        return df
    
    def train_model(self):
        # Load dataset
        df = self.load_dataset()
        
        df = df.dropna(subset=['text', 'sentiment'])
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        df['sentiment_encoded'] = self.label_encoder.fit_transform(df['sentiment'])
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['sentiment_encoded'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Create pipeline with limited vocabulary size
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', SGDClassifier(loss='hinge', random_state=42))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Get all possible label indices
        labels = self.label_encoder.transform(self.label_encoder.classes_)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("Model Performance:")
        print(classification_report(
            y_test, 
            y_pred, 
            labels=labels,  
            target_names=self.label_encoder.classes_
        ))
        
        return self
    
    def predict(self, sentences=None):
        """Demonstrate predictions, optionally with interactive input"""
        print("\n=== Semantic Analysis Demo ===")
        print("Type a sentence to analyze its sentiment.")
        print("Type 'exit' to end the demo.\n")
        
        if sentences is None:
            while True:
                print("\nEnter a sentence: ")
                sentence = input()
                if sentence.lower() == 'exit':
                    print("\nEnded")
                    break
                
                # Preprocess
                processed = self.preprocess_text(sentence)
                
                # Predict
                prediction = self.model.predict([processed])[0]
                predicted_label = self.label_encoder.inverse_transform([prediction])[0]
                
                print("\nAnalysis Result:")
                print("-" * 20)
                print(f"Input text: '{sentence}'")
                print(f"Sentiment: {predicted_label}")
                print("-" * 20)

# Main execution
def main():
    # Create demonstrator
    demonstrator = SemanticAnalysisDemonstrator()
    
    # Train model
    demonstrator.train_model()
    
    # Test predictions
    demonstrator.predict()

# Run the main function
if __name__ == "__main__":
    main()