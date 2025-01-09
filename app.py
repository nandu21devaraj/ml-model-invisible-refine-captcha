
from flask import Flask, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from flask_cors import CORS
app = Flask(__name__)
CORS(app)



# Load or train a basic machine learning model for bot detection
def train_model():
    # Example: Training data (for the sake of the demonstration)
    # Features might include: screen width, mouse movement frequency, etc.
    X= np.array([
        [1280, 720, 0.1, 1, 5],  # Human-like example
        [1366, 768, 0.5, 3, 10],  # Human-like example
        [1280, 720, 2.0, 0, 0],   # Bot-like example
        [800, 600, 5.0, 0, 1]     # Bot-like example
    ])
    
    # Labels: 1 for human, 0 for bot
    y = np.array([1, 1, 0, 0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # Save the model to a file
    joblib.dump(model, 'bot_detection_model.pkl')

# Load the trained model
def load_model():
    try:
        model = joblib.load('bot_detection_model.pkl')
    except FileNotFoundError:
        train_model()  # Train the model if not found
        model = joblib.load('bot_detection_model.pkl')
    return model

# Endpoint to analyze frontend data
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    print("Received Data from Frontend:")
    # Extract relevant environmental features from the request
    features = np.array([
        [
            data['screenWidth'],
            data['screenHeight'],
            len(data['mouseMovements']) / (data['interactionTime'] / 1000),  # Mouse movement frequency per second
            len(data['keyPresses']) / (data['interactionTime'] / 1000),  # Key press frequency per second
            len(data['mouseMovements'])  # Total mouse movement events
        ]
    ])

    # Load the pre-trained model
    model = load_model()

    # Predict if the user is a bot (0) or a human (1)
    is_bot = model.predict(features)[0]
    if(is_bot==0):
        print("Human");
    else:
        print("Bot");

    # Respond with the result
    return jsonify({'isBot': int(is_bot == 1)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)