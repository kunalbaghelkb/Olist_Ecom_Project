import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import silhouette_score, r2_score, accuracy_score, recall_score

def train_delivery_regression(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"📊 Regression R2 Score: {r2:.4f}")
    joblib.dump(model, 'models/delivery_time_model.pkl')
    return model

def train_late_classification(X_train, y_train, X_test, y_test):
    # Balanced weights logic from your notebook
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    
    preds = model.predict(X_test)
    print(f"📊 Classification Recall: {recall_score(y_test, preds):.4f}")
    joblib.dump(model, 'models/delivery_late_model.pkl')
    return model

def train_segmentation(rfm_scaled):
    kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    score = silhouette_score(rfm_scaled, clusters)
    print(f"📊 Segmentation Silhouette Score: {score:.4f}")
    joblib.dump(kmeans, 'models/customer_segmentation.pkl')
    return kmeans

def train_nlp_sentiment(X_text, y_sentiment):
    tfidf = TfidfVectorizer(max_features=5000)
    X_vec = tfidf.fit_transform(X_text)
    
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_vec, y_sentiment)
    
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    print("📊 NLP Model & Vectorizer Saved.")
    return model