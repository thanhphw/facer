#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <map>  // Include the map header
using namespace cv;

int main() {
    // Path to the face cascade XML file for face detection
    String face_cascade_path = "haarcascade_frontalface_default.xml";

    // Path to the directory containing the training images
    String training_dir = "training_data";

    // Create the LBPH face recognizer
    Ptr<cv::face::LBPHFaceRecognizer> recognizer = cv::face::LBPHFaceRecognizer::create();

    // Load the trained model
    recognizer->read("trained_model.xml");

    // Load the face cascade classifier
    CascadeClassifier face_cascade;
    face_cascade.load(face_cascade_path);

    // Initialize the camera
    VideoCapture camera(0);
    camera.set(CAP_PROP_FRAME_WIDTH, 640);   // Width
    camera.set(CAP_PROP_FRAME_HEIGHT, 480);  // Height

    // Font settings for displaying text on the image
    int font = FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.0;
    Scalar font_color(0, 255, 0);  // Green

    // Mapping between labels and names
    std::map<int, std::string> label_name_map;
    label_name_map[0] = "Phuong";
    label_name_map[1] = "Phuong";
    label_name_map[2] = "Phuong";
    label_name_map[3] = "Phuong";
    label_name_map[4] = "Phuong";
    label_name_map[5] = "Phuong";
    label_name_map[6] = "Phuong";
    label_name_map[7] = "Phuong";
    label_name_map[8] = "Hop";
    label_name_map[9] = "Hop";
    label_name_map[10] = "Hop";
    label_name_map[11] = "Hop";
    label_name_map[12] = "Hop";
    label_name_map[13] = "Hop";
    label_name_map[14] = "Hop";
    label_name_map[15] = "Hop";

    // Add more mappings as per your training data

    while (true) {
        // Capture frame-by-frame from the camera
        Mat frame;
        camera >> frame;

        // Convert the frame to grayscale for face detection
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces in the frame
        std::vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // Process each detected face
        for (const auto& face : faces) {
            // Extract the face region of interest (ROI)
            Mat face_roi = gray(face);

            // Recognize the face using the LBPH recognizer
            int label = -1;
            double confidence = 0.0;
            recognizer->predict(face_roi, label, confidence);
            std::cout << label << std::endl;
            // Get the name corresponding to the label
            std::string name = "Unknown";
            if (label_name_map.find(label) != label_name_map.end()) {
                name = label_name_map[label];
            }

            // Display the predicted name and confidence level
            std::string text = (confidence > 85 && confidence < 99) ?
                format("%d %s | %.2f", label, name.c_str(), confidence) :
                "Unknown";
            putText(frame, text, Point(face.x, face.y - 10), font, font_scale, font_color, 2, LINE_AA);

            // Draw a rectangle around the face
            rectangle(frame, face, Scalar(0, 255, 0), 2);
        }

        // Display the resulting frame
        imshow("Face Recognition", frame);

        // Exit the loop if 'q' is pressed
        if (waitKey(1) == 'q') {
            break;
        }
    
    }

    // Release the camera and close any open windows
    camera.release();
    destroyAllWindows();

    return 0;
}
