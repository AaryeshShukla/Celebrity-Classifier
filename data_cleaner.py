import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None:   # safety check
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return None

# Paths
path_to_data = "model\\dataaset"
path_to_crop_data = "model\\dataaset\\cropped"

# Gather folders
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

# Reset cropped folder
if os.path.exists(path_to_crop_data):
    shutil.rmtree(path_to_crop_data)
os.mkdir(path_to_crop_data)

# Output dict
cropped_image_dirs = []
celebrity_file_names_dict = {}

# Loop over celebrity folders
for img_dir in img_dirs:
    celebrity_name = os.path.basename(img_dir)
    celebrity_file_names_dict[celebrity_name] = []

    cropped_folder = os.path.join(path_to_crop_data, celebrity_name)
    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder)
        cropped_image_dirs.append(cropped_folder)
        print("Generating cropped images in folder:", cropped_folder)

    # Loop over images inside celebrity folder
    image_count = 0  # counter per celebrity
    for entry in os.scandir(img_dir):
        if not entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # skip non-images

        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            image_count += 1
            # Rename image as <celebrity_name>_<number>.jpg
            save_filename = f"{celebrity_name}_{image_count}.jpg"
            save_path = os.path.join(cropped_folder, save_filename)
            cv2.imwrite(save_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(save_path)
