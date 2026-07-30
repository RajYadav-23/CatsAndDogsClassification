import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = tf.keras.models.load_model('cat_dog_model.keras')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = tf.keras.utils.load_img(filepath, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = 'Dog' if prediction > 0.5 else 'Cat'
    confidence = float(prediction) if label == 'Dog' else float(1 - prediction)

    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 2),
        'image_url': f'/static/uploads/{filename}'
    })

if __name__ == '__main__':
    app.run(debug=True)
