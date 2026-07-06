/*#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>
#include <vector>

int main() {
    // --- 1. CHArUco BOARD CONFIGURATION ---
    // Define the structural parameters of your printed calibration board
    int squaresX = 8;               // Number of chessboard squares along the X-axis
    int squaresY = 6;               // Number of chessboard squares along the Y-axis
    float squareLength = 0.030f;    // Length of a square side (e.g., 0.030 meters / 30mm)
    float markerLength = 0.022f;    // Length of an ArUco marker side (e.g., 0.022 meters / 22mm)
    
    // Select the ArUco dictionary matching your target
    cv::aruco::PredefinedDictionaryType dictId = cv::aruco::DICT_6X6_250;
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictId);
    
    // Instantiate the ChArUco board layout configuration
    cv::aruco::CharucoBoard board(cv::Size(squaresX, squaresY), squareLength, markerLength, dictionary);

    // --- 2. STORAGE DATASTRUCTURES ---
    std::vector<std::vector<cv::Point2f>> allCharucoCorners;
    std::vector<std::vector<int>> allCharucoIds;
    cv::Size imageSize;

    // --- 3. IMAGES / DATA ACQUISITION POOL ---
    // Populate this list with paths to your snapshot image files
    std::vector<std::string> imageFiles = { "img_01.jpg", "img_02.jpg", "img_03.jpg" }; 

    if (imageFiles.empty()) {
        std::cerr << "Error: The calibration image collection is empty." << std::endl;
        return -1;
    }

    // --- 4. DETECTOR INITIALIZATION (OpenCV 4.x Pipeline) ---
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::RefineParameters refineParams;
    
    // CharucoDetector coordinates both marker collection and corner interpolation pipelines
    cv::aruco::CharucoDetector detector(board, cv::aruco::CharucoParameters(), detectorParams, refineParams);

    std::cout << "Processing collected viewpoints..." << std::endl;

    for (const auto& filePath : imageFiles) {
        cv::Mat image = cv::imread(filePath);
        if (image.empty()) {
            std::cerr << "Warning: Could not read target image: " << filePath << std::endl;
            continue;
        }

        imageSize = image.size();
        std::vector<cv::Point2f> charucoCorners;
        std::vector<int> charucoIds;

        // Detect underlying ArUco IDs and automatically interpolate chessboard structural intersections
        detector.detectBOARD(image, charucoCorners, charucoIds);

        // Accept frames providing enough overlapping constraints for calibration
        if (charucoIds.size() >= 4) {
            allCharucoCorners.push_back(charucoCorners);
            allCharucoIds.push_back(charucoIds);
        } else {
            std::cerr << "Warning: Insufficient features found in: " << filePath << std::endl;
        }
    }

    if (allCharucoCorners.empty()) {
        std::cerr << "Error: No valid ChArUco views were detected. Calibration canceled." << std::endl;
        return -1;
    }

    // --- 5. EXECUTE CAMERA CALIBRATION ---
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;
    int calibrationFlags = 0; // Default pinhole options. Standard cv::CALIB_flags apply here.

    std::cout << "Running mathematical optimization workspace..." << std::endl;
    
    double reprojectionError = cv::aruco::calibrateCameraCharuco(
        allCharucoCorners,
        allCharucoIds,
        board,
        imageSize,
        cameraMatrix,
        distCoeffs,
        rvecs,
        tvecs,
        std::noArray(), // Optional intrinsic standard deviations
        std::noArray(), // Optional extrinsic standard deviations
        std::noArray(), // Optional per-view reprojection errors
        calibrationFlags
    );

    // --- 6. DISPLAY AND EXPORT LOGS ---
    std::cout << "\n=== Calibration Process Concluded ===" << std::endl;
    std::cout << "Final Reprojection Error (RMS): " << reprojectionError << " pixels" << std::endl;
    std::cout << "\nIntrinsic Camera Matrix (K):\n" << cameraMatrix << std::endl;
    std::cout << "\nLens Distortion Coefficients (D):\n" << distCoeffs << std::endl;

    // Export internal properties to structural storage file
    cv::FileStorage fs("camera_calibration.xml", cv::FileStorage::WRITE);
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "reprojection_error" << reprojectionError;
    fs.release();
    
    std::cout << "\nParameters successfully written to disk ('camera_calibration.xml')." << std::endl;
    return 0;
}
*/
