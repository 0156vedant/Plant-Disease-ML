from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from flask_cors import CORS

# Load the pre-trained model
model = tf.keras.models.load_model('trained_model.keras')

# Define the class names
class_name = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

app = Flask(__name__)
CORS(app)
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load the image file as an RGB image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Preprocess the image (convert to array, resize, etc.)
        img = np.array(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert RGB to BGR (if required)
        image_resized = cv2.resize(img, (128, 128))  # Resize as per the model input requirement
        image_arr = tf.keras.preprocessing.image.img_to_array(image_resized)
        image_arr = np.expand_dims(image_arr, axis=0)  # Convert single image to batch

        # Perform the prediction
        prediction = model.predict(image_arr)
        result_index = np.argmax(prediction)
        model_prediction = class_name[result_index]

        # Return the prediction result
        return jsonify({
            'prediction': model_prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
