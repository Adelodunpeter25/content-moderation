"""Spam classification service using machine learning."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

class SpamClassifier:
    """Machine learning-based spam text classifier."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = MultinomialNB()
        self.is_trained = False
        
    def train(self, texts: list[str], labels: list[int]) -> None:
        """Train the spam classifier with text samples and labels.
        
        Args:
            texts: List of text samples
            labels: List of labels (1 for spam, 0 for ham)
        """
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True
        
    def predict(self, text: str) -> tuple[bool, float]:
        """Predict if text is spam.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (is_spam, confidence_score)
        """
        if not self.is_trained:
            self._load_or_create_model()
        
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X)[0].max()
        
        return bool(prediction), float(confidence)
    
    def _load_or_create_model(self) -> None:
        """Load existing model or create one with sample data."""
        # Simple training data for demo
        spam_texts = [
            "Free money now click here",
            "You won a prize claim now",
            "Limited time offer act fast",
            "Urgent response needed"
        ]
        ham_texts = [
            "Meeting scheduled for tomorrow",
            "Thanks for your help",
            "How are you doing today",
            "Project update attached"
        ]
        
        texts = spam_texts + ham_texts
        labels = [1] * len(spam_texts) + [0] * len(ham_texts)
        
        self.train(texts, labels)