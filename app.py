from flask import Flask, render_template, request,flash,redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
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
    #Information to display
    description = {
        0: "No diabetic retinopathy was detected in the image.",
        1: "Mild diabetic retinopathy was detected in the image. This stage may not require immediate treatment, but regular eye exams are recommended to monitor the condition.",
        2: "Moderate diabetic retinopathy was detected in the image. This stage requires close monitoring and possible treatment to prevent vision loss.",
        3: "Severe diabetic retinopathy was detected in the image. This stage requires immediate medical attention and treatment to prevent further vision loss.",
        4: "Proliferative diabetic retinopathy was detected in the image. This is the most advanced stage of the condition and requires prompt treatment to prevent blindness."
    }
    treatment = {
        0: "Maintain the sugar level to prevent any damage to eyes in the future.",
        1: "Patient will need to have regular eye exams to monitor the condition.",
        2: "May involve laser treatment to seal off leaking blood vessels or injections of medication into the eye to reduce swelling.",
        3: "Laser treatment is often used to prevent the progression of the disease, reduce swelling in the macula, and promote the growth of new blood vessels.",
        4: "May require laser treatment or surgery to remove the scar tissue and prevent further bleeding and vision loss."
    }
    # Map the predicted class to a label
    if predicted_class == 0:
        label = 'Class 0'
    elif predicted_class == 1:
        label = 'Class 1'
    elif predicted_class == 2:
        label = 'Class 2'
    elif predicted_class == 3:
        label = 'Class 3'
    else:
        label = 'Class 4'
    # Render the results template with the predicted label
    return render_template('results.html', label=label, description=description[predicted_class],treatment=treatment[predicted_class])

if __name__ == '__main__':
    app.run(debug=True)
