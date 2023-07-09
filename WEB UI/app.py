import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('CNN_CIFAR.h5')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def classify_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((32, 32))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize image

    result = model.predict(img)[0]
    predicted_class = np.argmax(result)
    confidence = result[predicted_class]
    class_label = class_names[predicted_class]

    return class_label, confidence

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Save the uploaded image
        uploaded_file = request.files['file']
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(image_path)

        # Classify the uploaded image
        class_label, confidence = classify_image(image_path)

        return render_template('result.html',
                               image_path=image_path,
                               class_label=class_label,
                               confidence=confidence)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
