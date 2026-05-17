import re
import spacy

# Load the spaCy English model
# (If it fails to load, ensure you ran: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model for spaCy...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text using spaCy by:
    - Lowercase conversion
    - Removing punctuation and special characters
    - Removing English stopwords
    - Lemmatization
    - Stripping extra whitespace
    
    Args:
        text (str): The raw input text.
        
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase conversion
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Process the text with spaCy
    # We increase max_length slightly in case of very long tickets, though default is 1M
    doc = nlp(text)
    
    # Filter out stopwords, punctuation, and spaces, then lemmatize
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    
    # Join words back and strip extra whitespace
    cleaned_text = ' '.join(words).strip()
    
    return cleaned_text
