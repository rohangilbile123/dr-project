from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('diabetic_retinopathy_cnn.h5')

def preprocess_image(image):
    # Resize image to (224, 224)
    img = image.resize((224, 224))
    # Convert to grayscale
    img = img.convert('L')
    # Convert to numpy array
    img_array = img_to_array(img)
    # Normalize pixel values to be between 0 and 1
    img_array /= 255.
    # Add batch dimension and return
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    # Get uploaded image from user
 if request.method == "POST":
    img = Image.open(request.files['file'])
    # Preprocess the image
    img_array = preprocess_image(img)
    # Get the predicted class from the model
    class_probs = model.predict(img_array)[0]
    predicted_class = np.argmax(class_probs)
    # Map the predicted class to a label
    if predicted_class == 0:
        label = 'No DR'
    elif predicted_class == 1:
        label = 'Mild DR'
    elif predicted_class == 2:
        label = 'Moderate DR'
    elif predicted_class == 3:
        label = 'Severe DR'
    else:
        label = 'Proliferative DR'
    # Render the results template with the predicted label
    return render_template('results.html', label=label)

if __name__ == '__main__':
    app.run(debug=True)
