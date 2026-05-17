# Support Ticket Classification & Prioritization

## About the Author
**Name:** Charan Akula  
**Role:** Machine Learning Intern  

**What I Learned:**  
Through completing this task, I have gained hands-on experience in building a complete Natural Language Processing (NLP) pipeline. I learned how to clean and preprocess text data, convert textual information into numerical features using TF-IDF, and apply Machine Learning algorithms (Logistic Regression) to solve real-world operational problems like ticket classification and prioritization. This project provided valuable, practical exposure to building end-to-end ML systems for business operations.

## Project Overview
This project implements an end-to-end Machine Learning pipeline to classify customer support tickets into specific categories (Billing, Technical Issue, Account, General Query) and predict their priority (High, Medium, Low). It uses Natural Language Processing (NLP) techniques for text preprocessing and feature extraction (TF-IDF), followed by Logistic Regression models for classification.

## Dataset Used
The pipeline is designed to work with support ticket datasets containing `ticket_text`, `category`, and `priority`. A synthetic dataset generator (`generate_data.py`) is provided, which creates 1,000 realistic support tickets to train and evaluate the models.

## Clear Explanations

### How Tickets are Categorized
Tickets are categorized using a Machine Learning approach rather than simple rule-based keywords. 
1. **Text Cleaning:** Raw ticket text is cleaned by removing punctuation, making everything lowercase, and filtering out common English stopwords (e.g., "the", "is", "at").
2. **Feature Extraction:** The cleaned text is transformed into numerical vectors using **TF-IDF** (Term Frequency-Inverse Document Frequency). This highlights words that are frequent in a specific ticket but rare across all tickets (e.g., "invoice", "crash", "password").
3. **Classification Model:** A **Logistic Regression** model is trained on these numerical vectors. It learns which specific words and phrases (unigrams and bigrams) strongly correlate with categories like 'Billing', 'Technical Issue', 'Account', and 'General Query'. When a new ticket arrives, the model calculates the probability of each category and outputs the one with the highest likelihood.

### How Priority is Decided
Similar to categorization, priority (High, Medium, Low) is predicted using a dedicated **Logistic Regression** model.
1. The model uses the same preprocessed, TF-IDF vectorized text features.
2. During training, the model identifies language patterns associated with urgency. For instance, words like "crashing", "locked", or "immediately" might become strong signals for a "High" priority, whereas phrases like "business hours" or "feature suggestion" might signal a "Low" priority.
3. This allows the system to route urgent issues to human agents faster, reducing response times for critical problems.

### Evaluation Results and Insights
Both models are evaluated on a 20% hold-out test set using standard ML metrics:
- **Accuracy:** The overall percentage of correct predictions.
- **Precision, Recall, and F1-Score:** These metrics are calculated for each specific class to ensure the model doesn't just guess the most common class.
- **Confusion Matrix:** A heatmap is generated to visualize where the model gets confused (e.g., mistaking an 'Account' issue for a 'General Query'). 

*Insight:* Text-based classification using TF-IDF and Logistic Regression establishes a very strong baseline for operational support pipelines, proving that complex Deep Learning isn't always necessary for effective triage systems.

## File Structure
- `data/`: Contains the generated dataset (`tickets.csv`).
- `models/`: Stores the trained models and vectorizer (`.pkl` files).
- `outputs/`: Stores the confusion matrix heatmaps (`.png` files).
- `notebooks/`: Contains the Google Colab `.ipynb` notebook.
- `generate_data.py`: Script to generate the synthetic dataset.
- `preprocess.py`: Contains text cleaning and preprocessing functions.
- `train.py`: The main training pipeline.
- `predict.py`: Contains the inference function to predict new tickets.
- `evaluate.py`: Contains evaluation metrics and plotting functions.
- `requirements.txt`: Python dependencies.

## Installation & Usage

1. **Set up virtual environment & install requirements:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Generate the synthetic dataset:**
   ```bash
   python generate_data.py
   ```

3. **Train the models:**
   ```bash
   python train.py
   ```
   This will preprocess the data, train the classifiers, generate evaluation metrics (and save confusion matrices to `outputs/`), and save the models to `models/`.

4. **Run predictions on sample tickets:**
   ```bash
   python predict.py
   ```

## Example Predictions
```json
{
  "ticket": "I was charged twice for my subscription this month. Please refund the extra charge.",
  "predicted_category": "Billing",
  "predicted_priority": "High"
}
```
