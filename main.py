from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import os
import face_recognition
from flask_cors import CORS
import uuid

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

CORS(app)


# define the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "Robby Birham")
    return "Selamat Hello {}!".format(name)

# Step 4: Define the API route
@app.route('/predict', methods=['POST'])

def predict():
    # Retrieve the input image
    file = crop(request.files['image'])
    file_knowledge = crop(request.files['image_knowledge'])

    known_image = face_recognition.load_image_file(file_knowledge)
    unknown_image = face_recognition.load_image_file(file)


    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)

    # remove file
    os.remove(file)
    os.remove(file_knowledge)

    # convert bool to string
    if results:
        result = True
    else:
        result = False

    response = {
        'data': {
            'prediction': result
        },
        'status' : 'success'
    }
    return jsonify(response)

def crop(file):
    # generate a unique filename using a random number
    _, ext = os.path.splitext(file.filename)
    filename = str(uuid.uuid4()) + ext
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
     # load the image and convert it to grayscale
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in the image using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # crop the image to only show the face(s)
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite(filepath, face_img)
    return filepath

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))