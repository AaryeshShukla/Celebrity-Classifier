import os
import json
import joblib
import numpy as np
import cv2
import base64
from wavelets import w2d  # make sure this file is present

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Global variables
__class_name_to_number = {}
__class_number_to_name = {}
__model = None

# ------------------------------
# Load saved artifacts
# ------------------------------
def load_saved_artifacts():
    global __class_name_to_number, __class_number_to_name, __model

    print("Loading saved artifacts...")
    with open(os.path.join(ARTIFACTS_DIR, "class_dictionary.json"), "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    if __model is None:
        __model = joblib.load(os.path.join(ARTIFACTS_DIR, "saved_model.pkl"))

    print("Artifacts loaded.")

# ------------------------------
# Convert class number to name
# ------------------------------
def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

# ------------------------------
# Convert base64 string to cv2 image
# ------------------------------
def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# ------------------------------
# Crop faces with at least 2 eyes
# ------------------------------
def get_cropped_image_if_2_eyes(image_path=None, image_base64_data=None):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

# ------------------------------
# Classify image
# ------------------------------
def classify_image(image_base64_data=None, image_path=None):
    global __model, __class_number_to_name
    cropped_faces = get_cropped_image_if_2_eyes(image_path, image_base64_data)
    results = []

    for img in cropped_faces:
        # Resize and wavelet
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'haar', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))

        # Flatten and combine
        combined_img = np.hstack((
            scalled_raw_img.flatten(),
            scalled_img_har.flatten()
        ))
        final_input = combined_img.reshape(1, -1)

        # Predict
        prediction = __model.predict(final_input)[0]
        probas = __model.predict_proba(final_input)[0]
        all_probs = {__class_number_to_name[i]: float(np.round(probas[i], 2)) 
                     for i in range(len(probas))}

        results.append({
            'class': class_number_to_name(prediction),
            'class_probability': np.around(probas*100, 2).tolist(),
            'probabilities': all_probs,
            'class_dictionary': __class_name_to_number
        })

    return results
