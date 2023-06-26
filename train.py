import cv2
import os

# Directory containing the training images
train_dir = '/home/phuongtt47/Desktop/LBPH/Data/Phuong'

# Initialize LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Iterate over the training images and labels
train_images = []
labels = []

for label in os.listdir(train_dir):
    label_dir = os.path.join(train_dir, label)
    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        train_images.append(image)
        labels.append(int(label))

# Train the LBPH recognizer
recognizer.train(train_images, np.array(labels))

# Save the trained model
recognizer.save('lbph_trained_model.xml')