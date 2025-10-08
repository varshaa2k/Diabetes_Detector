from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image, ImageEnhance
import cv2

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model('nail_disease_detector.h5')


# Define class labels (customize based on your model)
class_labels = ['Diabetes', 'Healthy', 'Other']  # Adjust if needed

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/privacy-policy')
def privacy():
    return render_template('privacy_policy.html')
  # Add at the top with other imports

# Function to adjust brightness using CLAHE (adaptive histogram equalization)
def adjust_image_brightness(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))  # Match model input size
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])
    
    image_output = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    image_output = image_output / 255.0  # Normalize to [0,1] range
    return np.expand_dims(image_output, axis=0)  # Add batch dimension


@app.route('/diabetes-detector', methods=['GET', 'POST'])
def diabetes_detector():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Preprocess and predict
            img_array = adjust_image_brightness(filepath)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = class_labels[predicted_class]
            print(predicted_label)
            confidence = round(np.max(prediction) * 100, 2)

            return render_template('diabetes_detector.html', filename=file.filename, label=predicted_label, confidence=confidence)
    return render_template('diabetes_detector.html', filename=None, label=None, confidence=None)




# Route to display uploaded image dynamically
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'), code=301)


if __name__ == '__main__':
    app.run(debug=True)
