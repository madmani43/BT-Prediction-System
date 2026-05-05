import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
template_dir = os.path.join(project_root, "templates")
model_path = os.path.join(project_root, "models", "model.h5")
UPLOAD_FOLDER = os.path.join(project_root, "uploads")

app = Flask(__name__, template_folder=template_dir)

# Load model
try:
    model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER   

def predict_tumor(image_path):
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions)
        confidence_score = float(np.max(predictions))

        if class_labels[predicted_class_index] == 'notumor':
            result = "No Tumor Detected"
        else:
            result = f"Tumor: {class_labels[predicted_class_index]}"

        return result, confidence_score
    except Exception as e:
        raise e

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template('index.html', result="No file uploaded", confidence=None)
        
        file = request.files["file"]
        
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_location = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_location)

            try:
                result, confidence = predict_tumor(file_location)
                confidence_str = f'{float(confidence)*100:.2f}%'
            except Exception as e:
                result = f"Error: {str(e)}"
                confidence_str = "N/A"

            return render_template('index.html', result=result, confidence=confidence_str, file_path=f'/uploads/{filename}')
        else:
            return render_template('index.html', result="Please select a file", confidence=None)

    return render_template('index.html', result=None, confidence=None) 

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    print("Starting Flask app on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)