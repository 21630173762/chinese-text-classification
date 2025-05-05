from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

class TraditionalModel:
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer()
        self.model = self._create_model()
        
    def _create_model(self):
        if self.model_type == 'naive_bayes':
            return Pipeline([
                ('tfidf', self.vectorizer),
                ('clf', MultinomialNB())
            ])
        elif self.model_type == 'svm':
            return Pipeline([
                ('tfidf', self.vectorizer),
                ('clf', SVC(kernel='linear'))
            ])
        elif self.model_type == 'logistic':
            return Pipeline([
                ('tfidf', self.vectorizer),
                ('clf', LogisticRegression(max_iter=1000))
            ])
        elif self.model_type == 'random_forest':
            return Pipeline([
                ('tfidf', self.vectorizer),
                ('clf', RandomForestClassifier(n_estimators=100))
            ])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, texts, labels):
        self.model.fit(texts, labels)
        
    def predict(self, texts):
        return self.model.predict(texts)
        
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)
        
    def get_params(self):
        return self.model.get_params()
        
    def set_params(self, **params):
        self.model.set_params(**params) 