import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocess import load_and_merge_data, feature_engineering
from src.models import train_delivery_regression, train_segmentation, train_nlp_sentiment, train_late_classification
from src.preprocess_nlp import clean_text

def run_pipeline():
    print("Starting Olist Master Pipeline...")
    os.makedirs('models', exist_ok=True)

    # DATA PREPARATION
    raw_df = load_and_merge_data()
    df = feature_engineering(raw_df)

    # 2. FEATURE SELECTION & ENCODING
    features = ['freight_value','product_weight_g','price','customer_state','product_category_name_english']
    X = df[features]
    X = pd.get_dummies(X, columns=['customer_state','product_category_name_english'], drop_first=True)
    X = X.fillna(X.median())
    
    # Regression & Classification
    y_reg = df.delivery_days
    y_class = df.is_late
    
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

    train_delivery_regression(X_train, y_train_reg, X_test, y_test_reg)
    train_late_classification(X_train, y_train_class, X_test, y_test_class)

    # CUSTOMER SEGMENTATION (RFM)
    print("⏳ Processing RFM Segmentation...")
    max_date = df['order_purchase_timestamp'].max()
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x:(max_date - x.max()).days,
        'order_id': 'count',
        'price': 'sum'
    })
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    import joblib
    joblib.dump(scaler, 'models/rfm_scaler.pkl')
    train_segmentation(rfm_scaled)

    # NLP SENTIMENT
    print("Training NLP Sentiment Engine...")
    df_nlp = df[df.review_comment_message.notnull()].copy()
    df_nlp['sentiment'] = df_nlp.review_score.apply(lambda x: 1 if x > 2 else 0)
    # Note: Ensure clean_text is imported correctly
    df_nlp['cleaned_review'] = df_nlp.review_comment_message.apply(clean_text)
    train_nlp_sentiment(df_nlp['cleaned_review'], df_nlp['sentiment'])

    print("All Artifacts saved!")

if __name__ == "__main__":
    run_pipeline()