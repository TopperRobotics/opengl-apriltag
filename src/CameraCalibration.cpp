#include "CameraCalibration.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <filesystem>
#include <chrono>

// CHArUco board parameters
int squaresX = 8;               // Number of chessboard squares along the X-axis
int squaresY = 6;               // Number of chessboard squares along the Y-axis
float squareLength = 0.030f;    // Length of a square side (e.g., 0.030 meters / 30mm)
float markerLength = 0.022f;    // Length of an ArUco marker side (e.g., 0.022 meters / 22mm)

std::vector<std::string> calibrationImages = {}; // to be populated with image file paths
std::string calibrationImagePath = "";

std::vector<float> cameraMatrix = {}; // fx, fy, cx, cy
std::vector<float> distCoeffs = {};

CameraCalibration::CameraCalibration() {
}

CameraCalibration::~CameraCalibration() {
    // Destructor
}

void CameraCalibration::setBoardParameters(int squaresX, int squaresY, float squareLength, float markerLength){
    this->squaresX = squaresX;
    this->squaresY = squaresY;
    this->squareLength = squareLength;
    this->markerLength = markerLength;
}

void CameraCalibration::setCalibrationImagePath(std::string path){
    this->calibrationImagePath = path;
}

void CameraCalibration::enumerateCalibrationImagesFromImagePath(){
    if(std::filesystem::exists(calibrationImagePath) && std::filesystem::is_directory(calibrationImagePath)){
        for (const auto& entry : std::filesystem::directory_iterator(targetDir)) {
                if (entry.is_regular_file()) {
                    std::cout << "File: " << entry.path() << "\n";
                    calibrationImages.push_back(entry.path());
                }
            }
    }
}

// true if successful, false if not
bool CameraCalibration::calibrateCamera(){
    cv::aruco::PredefinedDictionaryType dictId = cv::aruco::DICT_6X6_250;
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictId);
    cv::aruco::CharucoBoard board(cv::Size(squaresX, squaresY), squareLength, markerLength, dictionary);

    std::vector<std::vector<cv::Point2f>> allCharucoCorners;
    std::vector<std::vector<int>> allCharucoIds;
    cv::Size imageSize;

    enumerateCalibrationImagesFromImagePath();

    if(calibrationImages.empty()){
        std::cerr << "Error: No calibration images exist" << std::endl;
        return false;
    }

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::RefineParameters refineParams;
    
    // CharucoDetector coordinates both marker collection and corner interpolation pipelines
    cv::aruco::CharucoDetector detector(board, cv::aruco::CharucoParameters(), detectorParams, refineParams);

    std::cout << "Processing collected viewpoints..." << std::endl;

    for (const auto& filePath : calibrationImages) {
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
        return false;
    }

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

    this->cameraMatrix.push_back(extractValueFromMat(cameraMatrix, 0, 0)); // fx
    this->cameraMatrix.push_back(extractValueFromMat(cameraMatrix, 1, 1)); // fy
    this->cameraMatrix.push_back(extractValueFromMat(cameraMatrix, 0, 2)); // cx
    this->cameraMatrix.push_back(extractValueFromMat(cameraMatrix, 1, 2)); // cy

    for(int i = 0; i < 5; i++){
        this->distCoeffs.push_back(extractValueFromMat(distCoeffs, 0, i));
    }

    auto now = std::chrono::system_clock::now();
    
    // Convert the time point to a duration since the epoch (Jan 1, 1970)
    auto duration = now.time_since_epoch();
    
    // Cast the duration explicitly into seconds and extract the integer count
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

    cv::FileStorage fs("camera_calibration_" + seconds + ".xml", cv::FileStorage::WRITE);
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "reprojection_error" << reprojectionError;
    fs.release();

    return true;
}

float extractValueFromMat(cv::Mat mat, int row, int col){
    return mat.at<float>(row, col);
}

std::vector<float> CameraCalibration::getCalibratedCameraMatrix(){
    return cameraMatrix;
}

std::vector<float> CameraCalibration::getCalibratedDistortionCoefficients(){
    return distCoeffs;
}