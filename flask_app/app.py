from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # type: ignore
import os
import tensorflow as tf
import requests
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
import numpy as np

app = Flask(__name__)  # Fixed here: _name_ to __name__
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model
model = tf.keras.models.load_model('Team1.h5')

# Define the labels for the model
labels = {
    0: 'chilli pepper', 1: 'mango', 2: 'pomegranate', 3: 'tomato', 4: 'cabbage',
    5: 'bell pepper', 6: 'eggplant', 7: 'garlic', 8: 'beetroot', 9: 'ginger',
    10: 'orange', 11: 'sweetpotato', 12: 'jalepeno', 13: 'raddish', 14: 'kiwi',
    15: 'lettuce', 16: 'carrot', 17: 'onion', 18: 'cauliflower', 19: 'watermelon',
    20: 'pineapple', 21: 'spinach', 22: 'pear', 23: 'peas', 24: 'turnip',
    25: 'sweetcorn', 26: 'potato', 27: 'corn', 28: 'grapes', 29: 'capsicum'
}

# Route for homepage
@app.route('/')
def index():
    # Ensure index.html is in the 'templates' folder
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    print("Received a request at /predict")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'})

    # Save the uploaded file
    file_path = os.path.join('static/uploads', file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)

    # Predict the image category
    predicted_label = predict_image(file_path)

    # Get the nutrition data based on the predicted label
    nutrition_data = get_nutrition_info(predicted_label)

    # Prepare the result
    result = {
        'predicted_label': predicted_label,
        'nutrition_data': nutrition_data if isinstance(nutrition_data, dict) else {'error': nutrition_data}
    }

    return jsonify(result)

# Function to predict the image label using the pre-trained model
def predict_image(file_path):
    try:
        img = load_img(file_path, target_size=(224, 224))  # Adjust target size according to your model input size
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        prediction = model.predict(img_array)  # Predict using the model
        predicted_class = np.argmax(prediction)  # Get the predicted class index
        predicted_label = labels[predicted_class]  # Map index to label
        return predicted_label
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Function to fetch nutrition information from the external API
def get_nutrition_info(predicted_label):
    try:
        api_url = f'https://api.api-ninjas.com/v1/nutrition?query={predicted_label}'
        response = requests.get(api_url, headers={'X-Api-Key': 'FJCuuMBu83nEaI7CGroR8A==z88oB9CslsPk9p80'})

        if response.status_code == requests.codes.ok:
            nutrition_data = json.loads(response.text)
            return nutrition_data[0] if nutrition_data else {"message": "No nutrition information available."}
        else:
            return {"error": f"API error: {response.status_code} - {response.text}"}
    except requests.RequestException as e:
        return f"Request failed: {str(e)}"

if __name__ == '__main__':  # Fixed here: _name_ to __name__
    app.run(debug=True)
