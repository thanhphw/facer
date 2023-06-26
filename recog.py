import cv2

# Load the face cascade XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained LBPH model
recognizer.read('lbph_trained_model.xml')

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        face_roi = gray[y:y+h, x:x+w]

        # Perform face recognition on the ROI
        label, confidence = recognizer.predict(face_roi)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the recognized label and confidence level
        text = f"Label: {label}, Confidence: {confidence}"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()