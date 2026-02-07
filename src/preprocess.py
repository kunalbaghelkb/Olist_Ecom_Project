import re
import string

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