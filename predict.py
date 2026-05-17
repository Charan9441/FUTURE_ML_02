import joblib
import os
from preprocess import preprocess_text

def predict_ticket(ticket_text: str, models_dir='models') -> dict:
    """
    Predicts the category and priority of a given support ticket.
    
    Args:
        ticket_text (str): The raw text of the ticket.
        models_dir (str): Directory where the models are saved.
        
    Returns:
        dict: A dictionary containing original text and predicted category.
    """
    try:
        vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
        cat_model = joblib.load(os.path.join(models_dir, 'category_model.pkl'))
    except FileNotFoundError:
        print("Models not found. Please run train.py first.")
        return {}
        
    # Preprocess text
    cleaned_text = preprocess_text(ticket_text)
    
    if not cleaned_text:
        return {
            "ticket": ticket_text,
            "predicted_category": "Unknown"
        }
        
    # Transform text using loaded vectorizer
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict
    pred_cat = cat_model.predict(text_tfidf)[0]
    
    return {
        "ticket": ticket_text,
        "predicted_category": pred_cat
    }

if __name__ == "__main__":
    sample_tickets = [
        "I was charged twice for my subscription this month. Please refund the extra charge.",
        "The application keeps crashing every time I try to upload a file.",
        "What are your business hours? I need to call support.",
        "How do I change the email address associated with my account?"
    ]
    
    print("Testing predictions with sample tickets:\n")
    for ticket in sample_tickets:
        result = predict_ticket(ticket)
        print(f"Ticket: {result['ticket']}")
        print(f"Predicted Category: {result.get('predicted_category')}")
        print("-" * 50)
