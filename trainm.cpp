
#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <dirent.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace cv;

int main() {
    std::string training_dir = "/home/phuongtt47/Desktop/LBPH/Data";

    // Vector to store the training images
    std::vector<Mat> images;

    // Vector to store the corresponding labels for each image
    std::vector<int> labels;
    std::map<int, std::string> label_name_map;
    // Read images from the training directory and assign labels
    DIR* dir;
    struct dirent* entry;
    if ((dir = opendir(training_dir.c_str())) != nullptr) {
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_DIR && std::string(entry->d_name) != "." && std::string(entry->d_name) != "..") {
                std::string person_dir = training_dir + "/" + entry->d_name;
                DIR* person_dir_ptr;
                struct dirent* person_entry;

                if ((person_dir_ptr = opendir(person_dir.c_str())) != nullptr) {
                    while ((person_entry = readdir(person_dir_ptr)) != nullptr) {
                        if (person_entry->d_type == DT_REG) {  // Check if it's a regular file
                            std::string image_path = person_dir + "/" + person_entry->d_name;
                            //std::cout << "Loading image: " << image_path << std::endl;
                            // Load the image in grayscale
                            Mat image = imread(image_path, IMREAD_GRAYSCALE);
                            if (image.empty()) {
                                std::cerr << "Error loading image: " << image_path << std::endl;
                                continue;
                            }
                            
                            // Extract the label from the folder name


                            // Add the image and label to the training vectors
                            images.push_back(image);
                            std::string filename = person_entry->d_name;
                            std::regex regex("[0-9]+");  // Regular expression pattern to match digits
                            std::smatch match;
                            if (std::regex_search(filename, match, regex)) {
                                int label = std::stoi(match.str());  // Convert the matched string to an integer
                                labels.push_back(label);  // Add the label to the labels vector
                                //std::cout << "Extracting label from filename: " << person_dir << filename << std::endl;
                                label_name_map[label] = person_dir;
                                std::cout << label << ":" << person_dir << std::endl;
                            }
                             else {
                                std::cerr << "Error extracting label from filename: " << filename << std::endl;
                                continue;  // Skip this image if label extraction fails
                            }
                        }
                    }
                    closedir(person_dir_ptr);
                } else {
                    std::cerr << "Error opening directory: " << person_dir << std::endl;
                    continue;
                }
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory: " << training_dir << std::endl;
        return 1;
    }

    Mat labelsMat(labels.size(), 1, CV_32SC1);
    for (size_t i = 0; i < labels.size(); i++) {
        labelsMat.at<int>(i) = labels[i];
        std::cout << labels[i] << " : " << i << std::endl;
    }

    Ptr<cv::face::LBPHFaceRecognizer> recognizer = cv::face::LBPHFaceRecognizer::create();
    // Train the recognizer with the training data
    recognizer->train(images, labelsMat);
    // Save the trained model
    recognizer->save("trained_model.xml");
    return 0;
}
