from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
import os
import face_recognition
import cv2
import threading
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///raman.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = '75eec60cdb48ca9c40c93d84fcd532fdc6e61a648ee4a35f'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'gramankishore@gmail.com'  
app.config['MAIL_PASSWORD'] = 'tijr mlzz oyut rmvp'  
mail = Mail(app)

db = SQLAlchemy(app)

from flask import send_from_directory
def load_known_faces():
    path = 'uploads'
    images = []
    class_names = []
    known_encodes = []

    for img in os.listdir(path):
        image = cv2.imread(f'{path}/{img}')
        images.append(image)
        class_names.append(os.path.splitext(img)[0])

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            known_encodes.append(encode)
        except IndexError:
            print("Face not found in one of the training images. Skipping...")

    return known_encodes, class_names

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

class MissingPerson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    address = db.Column(db.String(200), nullable=False)
    aadhaar_number = db.Column(db.String(20), nullable=False)
    missing_date = db.Column(db.String(20), nullable=False)
    missing_location = db.Column(db.String(200), nullable=False)
    photo_path = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/file-complaint', methods=['GET', 'POST'])
def file_complaint_route():
    if request.method == 'POST':
        form_data = request.form
        try:
            photo = request.files['photo']
            if photo:
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo.filename)
                photo.save(photo_path)

                new_complaint = MissingPerson(
                    first_name=form_data['fname'],
                    last_name=form_data['lname'],
                    age=form_data['age'],
                    address=form_data['address'],
                    aadhaar_number=form_data.get('aadhaar_number', ''),
                    missing_date=form_data['missing_date'],
                    missing_location=form_data['missing_location'],
                    photo_path=photo_path
                )
                db.session.add(new_complaint)
                db.session.commit()
                flash('Complaint filed successfully!')
                return redirect(url_for('index'))
        except Exception as e:
            flash(f'Error: {str(e)}')
            return redirect(url_for('file_complaint_route'))
    return render_template('complaint.html')

@app.route('/complaint-list')
def complaint_list():
    complaints = MissingPerson.query.all()
    return render_template('list.html', complaints=complaints)

@app.route('/complaint/<int:complaint_id>', methods=['GET', 'POST'])
def complaint_details(complaint_id):
    complaint = MissingPerson.query.get_or_404(complaint_id)
    if request.method == 'POST':
        found_photo = request.files['found_photo']
        if found_photo:
            found_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], found_photo.filename)
            found_photo.save(found_photo_path)
            if compare_faces(complaint.photo_path, found_photo_path):
                send_email(complaint)
                flash('Match found! An email has been sent to the complainant.')
            else:
                flash('No match found.')
            return redirect(url_for('complaint_list'))
    return render_template('complaint_details.html', complaint=complaint, complaint_id=complaint_id)

def compare_faces(known_image_path, unknown_image_path):
    # Check if files exist
    if not os.path.exists(known_image_path) or not os.path.exists(unknown_image_path):
        print("Error: One or both image files do not exist.")
        return False
    
    # Load images using OpenCV
    known_image = cv2.imread(known_image_path)
    unknown_image = cv2.imread(unknown_image_path)

    # Ensure images are in RGB format
    known_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
    unknown_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)

    # Detect and encode faces
    known_encodings = face_recognition.face_encodings(known_image)
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    # Handle cases where no face is detected
    if not known_encodings or not unknown_encodings:
        print("Error: No face detected in one or both images.")
        return False

    # Compare the first detected face in each image
    return face_recognition.compare_faces([known_encodings[0]], unknown_encodings[0])[0]

def send_email(complaint):
    msg = Message("Missing Person Found",
                  sender=app.config['MAIL_USERNAME'],
                  recipients=[app.config['MAIL_USERNAME']])  # Change to complainant's email if available
    msg.body = f"""We are pleased to inform you that the missing person missing from {complaint.missing_date} has been found.
Here are the details:
 - Name: {complaint.first_name} {complaint.last_name}
 - Date and Time of Sighting: {complaint.missing_date}
 - Location: {complaint.missing_location}

We understand the relief this news must bring to you. If you have any further questions, please do not hesitate to reach out.

Thank you for your cooperation.
"""
    mail.send(msg)

@app.route('/surveillance', methods=['GET'])
def surveillance():
    return render_template('surveillance.html')

def detect_faces_in_realtime():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
def detect_faces_in_realtime():
    known_encodes, class_names = load_known_faces()
    cap = cv2.VideoCapture(0)
    scale = 0.25
    box_multiplier = 1 / scale

    while True:
        success, img = cap.read()
        if not success:
            break

        Current_image = cv2.resize(img, (0, 0), None, scale, scale)
        Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(Current_image, model='hog')
        face_encodes = face_recognition.face_encodings(Current_image, face_locations)

        for encodeFace, faceLocation in zip(face_encodes, face_locations):
            matches = face_recognition.compare_faces(known_encodes, encodeFace, tolerance=0.6)
            faceDis = face_recognition.face_distance(known_encodes, encodeFace)
            matchIndex = np.argmin(faceDis)
            name = "Unknown"

            if matches[matchIndex]:
                name = class_names[matchIndex].upper()
                print(f'Match found: {name}')

            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = int(y1 * box_multiplier), int(x2 * box_multiplier), int(y2 * box_multiplier), int(x1 * box_multiplier)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Live Surveillance", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
@app.route('/start-surveillance', methods=['POST'])
def start_surveillance():
    threading.Thread(target=detect_faces_in_realtime, daemon=True).start()
    return jsonify({'status': 'surveillance started'})

if __name__ == '__main__':
    app.run(debug=True)
