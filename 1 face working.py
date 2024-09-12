import cv2s
import numpy as np

def train_face_recognizer(images, labels):
    # Create a face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the face recognizer with the provided images and labels
    face_recognizer.train(images, np.array(labels))

    return face_recognizer

def compare_faces(image, face_recognizer, gray_face, face_cascade):
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return False

    (x, y, w, h) = faces[0]

    # Extract face region of interest
    face_roi = gray_face[y:y+h, x:x+w]

    # Resize face image to a fixed size
    face_roi = cv2.resize(face_roi, (100, 100))

    # Compare the face with the provided recognizer
    label, confidence = face_recognizer.predict(face_roi)
    if confidence < 70:  # You can adjust the confidence threshold as needed
        return True
    else:
        return False

# Main code
given_image = "D:/python mini project/dataset/20230531_133131.jpg"

# Open the camera
cap = cv2.VideoCapture(0)

# Create a face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the given image and convert it to grayscale
given_img = cv2.imread(given_image)
given_gray = cv2.cvtColor(given_img, cv2.COLOR_BGR2GRAY)

# Train the face recognizer with the given image
face_recognizer = train_face_recognizer([given_gray], [1])

while True:
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the camera frame
    cv2.imshow('Camera', frame)

    # Check if the person in front of the camera matches the given image
    if compare_faces(frame, face_recognizer, gray_frame, face_cascade):
        print("kishore kumar !")
        break
    else :
        print("not match")
        break
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
