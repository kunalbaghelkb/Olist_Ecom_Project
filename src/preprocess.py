import re
import string
import pandas as pd
import numpy as np

def clean_text(text):
    """
    Cleans raw text by removing numbers, punctuation, and newlines.
    Used for Sentiment Analysis.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    
    return text.strip()

def load_and_merge_data(data_path='data/'):
    orders = pd.read_csv(f'{data_path}olist_orders_dataset.csv')
    items = pd.read_csv(f'{data_path}olist_order_items_dataset.csv')
    products = pd.read_csv(f'{data_path}olist_products_dataset.csv')
    translation = pd.read_csv(f'{data_path}product_category_name_translation.csv')
    reviews = pd.read_csv(f'{data_path}olist_order_reviews_dataset.csv')
    customers = pd.read_csv(f'{data_path}olist_customers_dataset.csv')

    df = pd.merge(orders, items, on='order_id', how='inner')
    df = pd.merge(df, products, on='product_id', how='left')
    df = pd.merge(df, translation, on='product_category_name', how='left')
    df = pd.merge(df, reviews, on='order_id', how='left')
    df = pd.merge(df, customers, on='customer_id', how='left')
    return df

def feature_engineering(df):
    date_cols = ['order_purchase_timestamp','order_delivered_customer_date','order_estimated_delivery_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df = df[df.order_status == 'delivered'].copy()
    
    # Target Features
    df['delivery_days'] = (df.order_delivered_customer_date - df.order_purchase_timestamp).dt.days
    df['is_late'] = (df.order_delivered_customer_date > df.order_estimated_delivery_date).astype(int)
    df.product_category_name_english = df.product_category_name_english.fillna('Unknown')
    
    # Filter 99th percentile outliers
    q99 = df.delivery_days.quantile(0.99)
    df = df[df.delivery_days <= q99].copy()
    
    return df