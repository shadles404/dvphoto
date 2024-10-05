import os
from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import dlib
from werkzeug.utils import secure_filename

# Set up Flask app
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load dlib's pre-trained facial landmark predictor
PREDICTOR_PATH = "C:/Users/Shadles/Desktop/trade/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Function to crop the image to 600x600 pixels with specified positions
def crop_image_to_600x600(image):
    # Define target crop dimensions
    target_height = 600
    target_width = 600

    # Calculate the cropping box
    height, width = image.shape[:2]

    # Ensure we fit the specified Y positions within the cropped area
    head_start_y = 30  # Head starts at Y position 30
    eye_start_y = 230
    neck_start_y = 416

    # Calculate the crop coordinates
    crop_start_y = head_start_y  # Starting from the head position
    crop_end_y = min(neck_start_y + (target_height - (neck_start_y - head_start_y)), height)
    crop_start_x = (width - target_width) // 2  # Center the crop horizontally
    crop_end_x = crop_start_x + target_width

    # Make sure to stay within the image bounds
    crop_start_y = max(crop_start_y, 0)
    crop_end_y = min(crop_end_y, height)

    # Perform cropping
    cropped_image = image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

    # Resize to ensure it's exactly 600x600
    final_image = cv2.resize(cropped_image, (target_width, target_height))

    return final_image

# Function to process the image
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Could not load image."

    # Crop the image to 600x600 pixels
    cropped_image = crop_image_to_600x600(image)

    # Convert to grayscale for facial landmark detection
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None, "No face detected in the image."

    for face in faces:
        landmarks = predictor(gray, face)
        return cropped_image, None  # No repositioning needed in this context

# Route for uploading and processing the image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image
            processed_image, error = process_image(filepath)
            if error:
                return error

            # Save the processed image
            processed_filename = f"processed_{filename}"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, processed_image)

            # Return the processed image for download
            return send_file(processed_filepath, as_attachment=True)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
