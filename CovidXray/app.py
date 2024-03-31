#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np


app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
model = load_model('Xray_Covid19/model.h5')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Function to perform binary image classification

# Function to perform binary image classification
def classify_image(file_path, threshold=0.5):
    img = image.load_img(file_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make a binary prediction
    result = model.predict(img_array)

    # Determine the label based on the threshold
    label = 'COVID-19' if result[0][0] > threshold else 'Normal'

    # Calculate the probability as a percentage
    probability = result[0][0] * 100

    # Include both label and probability in the result
    result = [{'label': label, 'probability': probability}]

    return result



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #print('upload_image filename: ' + filename)
        result = classify_image(file_path)
        
        flash('')
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename, result=result)
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run(debug=True)