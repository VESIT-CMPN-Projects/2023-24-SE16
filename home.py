from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import numpy as np
from keras.models import model_from_json
import firebase_admin
from firebase_admin import auth, credentials
import bcrypt
import requests


app = Flask(__name__)

cred = credentials.Certificate("gesturedetector-594c8-firebase-adminsdk-2xp5q-811b7f3651.json")
firebase_admin.initialize_app(cred)

json_file = open("Image48.json", "r")
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("Image48.h5")

label = ['IloveYou', 'blank', 'hello', 'house', 'middle', 'money', 'moon', 'namaste', 'practice', 'snake', 'thumbsup']

def extract_features(image):
    feature = np.array(image)
    feature = cv2.resize(feature, (48, 48))
    feature = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        success, frame = video_capture.read()
        
        if not success:
            break
        else:
            crop_frame = frame[40:300, 0:300]
            crop_frame = extract_features(crop_frame)
            
            prediction = model.predict(crop_frame)
            prediction_label = label[prediction.argmax()]

            accuracy = "{:.2f}".format(np.max(prediction) * 100)
            cv2.putText(frame, f'{prediction_label} {accuracy}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (30, 30), (300, 300), (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def first():
    return render_template('new.html')

@app.route('/object_detection')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email')
    password = request.form.get('password')
    password_confirm = request.form.get('password_confirm')

    if password != password_confirm:
        return jsonify({'error': 'Passwords do not match'}), 400

    try:
        user = auth.create_user(email=email, password=password)
        return redirect(url_for('home'))

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/login', methods=['POST'])
def login():
    try:
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Send a request to Firebase Authentication REST API for sign-in
        request_data = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        response = requests.post(
            "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyBUkLIk18ZLbH6RJQPkDODMeulhRk9rlNM",
            json=request_data
        )
        data = response.json()

        if 'error' in data:
            # If there's an error, return the error message
            return jsonify({'error': data['error']['message']}), 400
        else:
            # If successful, redirect to the home page
            return redirect(url_for('home'))

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
