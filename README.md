🌟 Celebrity Image Classifier

A Machine Learning powered web application that predicts which celebrity appears in an uploaded image.

This project covers the complete ML pipeline — from data collection and preprocessing to model training and deployment in a web interface.

🚀 Features

Upload an image and get celebrity prediction

Built using classical Machine Learning models

Custom dataset created and cleaned manually

Wavelet Transform used for advanced feature extraction

Clean web interface for user interaction

🧠 Machine Learning Pipeline
1️⃣ Data Collection

Images of celebrities were scraped and organized into labeled folders. Irrelevant and low-quality images were removed manually.

2️⃣ Image Preprocessing

Images resized to fixed dimensions

Wavelet transform applied to extract texture and edge features

Raw image pixels + wavelet features combined into a single feature vector

3️⃣ Model Training

The following models were trained and tuned using GridSearchCV:

Support Vector Machine (SVM)

Random Forest

Logistic Regression

The best-performing model was selected based on cross-validation accuracy.

4️⃣ Model Evaluation

Performance was evaluated using:

Confusion Matrix

Accuracy score

Classification report

🌐 Web Application

Users can upload an image through the web interface, and the system predicts the closest matching celebrity.

Frontend built with:

HTML

CSS

Bootstrap

Backend powered by:

Flask

Scikit-learn

OpenCV

🛠️ Tech Stack

Machine Learning & Backend

Python

OpenCV

NumPy

PyWavelets

Scikit-learn

Flask

Joblib

Frontend

HTML5

CSS3

Bootstrap

JavaScript

🎯 Supported Celebrities

Lionel Messi

Brad Pitt

Angelina Jolie

Jennifer Lawrence

Scarlett Johansson

Sydney Sweeney

(Add others if included)

▶️ How to Run This Project
git clone https://github.com/yourusername/celebrity-classifier.git
cd celebrity-classifier
pip install -r requirements.txt
python server.py


Then open your browser and go to:

http://127.0.0.1:5000

📌 Future Improvements

Add more celebrities and larger dataset

Improve accuracy using Deep Learning (CNN)

Add automatic face detection before classification

Deploy on cloud (Render / AWS / Heroku)

🙌 Author

Aaryesh Shukla
Machine Learning Enthusiast | Computer Science Student
