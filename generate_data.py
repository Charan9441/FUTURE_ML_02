import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(num_records=1000, output_path='data/tickets.csv'):
    categories = ['Billing', 'Technical Issue', 'Account', 'General Query']
    priorities = ['High', 'Medium', 'Low']
    
    # Templates for different categories
    billing_templates = [
        "I was charged twice for my subscription this month.",
        "My invoice for this month is incorrect, showing extra fees.",
        "Can I get a refund for the recent transaction? I didn't authorize it.",
        "How do I update my payment method for the next billing cycle?",
        "My credit card was declined but I have sufficient funds.",
        "There's an unrecognized charge of $50 on my statement from your company.",
        "Please cancel my subscription and refund my last payment."
    ]
    
    tech_templates = [
        "The application keeps crashing every time I try to upload a file.",
        "I'm getting a 500 Internal Server Error when accessing the dashboard.",
        "The software is completely unresponsive after the recent update.",
        "My synchronization is failing. It says 'connection timeout'.",
        "The API endpoints are returning 404 errors for my requests.",
        "I can't log in, the page just refreshes continuously.",
        "There's a significant lag when trying to load the report section."
    ]
    
    account_templates = [
        "I need to reset my password but I'm not getting the recovery email.",
        "How do I change the email address associated with my account?",
        "My account has been locked. Please unlock it immediately.",
        "I want to delete my account permanently.",
        "Can you merge my two accounts into one?",
        "I need to update my profile information but it gives an error.",
        "How can I add another user to my team account?"
    ]
    
    general_templates = [
        "What are your business hours?",
        "Do you have a feature for exporting data to PDF?",
        "Where can I find the documentation for the API?",
        "Are you planning to add dark mode anytime soon?",
        "How does the pricing tier work for enterprise customers?",
        "Do you offer discounts for non-profit organizations?",
        "I have a suggestion for a new feature."
    ]
    
    data = []
    
    for _ in range(num_records):
        category = random.choice(categories)
        
        # Priority logic: some correlations to make the ML model learn
        if category == 'Technical Issue':
            priority = np.random.choice(priorities, p=[0.6, 0.3, 0.1])
            text = random.choice(tech_templates)
        elif category == 'Billing':
            priority = np.random.choice(priorities, p=[0.5, 0.4, 0.1])
            text = random.choice(billing_templates)
        elif category == 'Account':
            priority = np.random.choice(priorities, p=[0.3, 0.5, 0.2])
            text = random.choice(account_templates)
        else:
            priority = np.random.choice(priorities, p=[0.05, 0.35, 0.6])
            text = random.choice(general_templates)
            
        # Add some random noise/variations
        noise_words = [" URGENT!", " Please help.", " ASAP.", " Thanks.", " Anyone there?", " Hello, ", ""]
        text = random.choice(noise_words) + text + random.choice(noise_words)
        
        data.append({
            'ticket_text': text.strip(),
            'category': category,
            'priority': priority
        })
        
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_records} synthetic tickets and saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
