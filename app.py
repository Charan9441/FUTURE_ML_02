from flask import Flask, render_template, request, jsonify
from predict import predict_ticket

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'ticket_text' not in data:
        return jsonify({'error': 'No ticket text provided'}), 400
        
    ticket_text = data['ticket_text']
    result = predict_ticket(ticket_text)
    
    if not result:
        return jsonify({'error': 'Models not found. Train the model first.'}), 500
        
    return jsonify({
        'predicted_category': result.get('predicted_category', 'Unknown')
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
