#include "CameraCalibration.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/charuco_detector.hpp> // opencv2/aruco/charuco_detector.hpp was moved
#include <filesystem>
#include <chrono>

static double extractValueFromMat(const cv::Mat& mat, int row, int col) {
    return mat.at<double>(row, col);
}

CameraCalibration::CameraCalibration() {
}

CameraCalibration::~CameraCalibration() {
}

void CameraCalibration::setBoardParameters(int sx, int sy, float sl, float ml) {
    squaresX = sx;
    squaresY = sy;
    squareLength = sl;
    markerLength = ml;
}

void CameraCalibration::setBoardParametersSX(int sx) {
    squaresX = sx;
}

void CameraCalibration::setBoardParametersSY(int sy) {
    squaresY = sy;
}

void CameraCalibration::setBoardParametersSL(float sl) {
    squareLength = sl;
}

void CameraCalibration::setBoardParametersML(float ml) {
    markerLength = ml;
}

void CameraCalibration::setCalibrationImagePath(std::string path) {
    calibrationImagePath = std::move(path);
}

void CameraCalibration::setCalibrationOutputPath(std::string path){
    calibrationOutputPath = std::move(path);
}

void CameraCalibration::clearCalibrationImagePath(){
    if (std::filesystem::exists(calibrationImagePath) && std::filesystem::is_directory(calibrationImagePath)) {
        for (const auto& entry : std::filesystem::directory_iterator(calibrationImagePath)) {
            if (entry.is_regular_file()) {
                std::cout << "Deleting File: " << entry.path() << "\n";
                if(std::filesystem::remove(entry.path())){
                    std::cout << "File deleted" << std::endl;
                } else {
                    std::cout << "File could not be deleted" << std::endl;
                }
                calibrationImages.push_back(entry.path().string());
            }
        }
    }
}

void CameraCalibration::enumerateCalibrationImagesFromImagePath() {
    if (std::filesystem::exists(calibrationImagePath) && std::filesystem::is_directory(calibrationImagePath)) {
        for (const auto& entry : std::filesystem::directory_iterator(calibrationImagePath)) {
            if (entry.is_regular_file()) {
                std::cout << "File: " << entry.path() << "\n";
                calibrationImages.push_back(entry.path().string());
            }
        }
    }
}

bool CameraCalibration::calibrateCamera() {
    cv::aruco::PredefinedDictionaryType dictId = cv::aruco::DICT_6X6_250;
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictId);
    cv::aruco::CharucoBoard board(cv::Size(squaresX, squaresY), squareLength, markerLength, dictionary);
    board.setLegacyPattern(false); // do not enable

    // NEW: accumulate matched 3D object points / 2D image points instead of raw corners/ids
    std::vector<cv::Mat> allObjPoints;
    std::vector<cv::Mat> allImgPoints;
    cv::Size imageSize;

    enumerateCalibrationImagesFromImagePath();

    if (calibrationImages.empty()) {
        std::cerr << "Error: No calibration images exist" << std::endl;
        return false;
    }

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::RefineParameters refineParams;
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

        detector.detectBoard(image, charucoCorners, charucoIds);

        if (charucoIds.size() >= 4) {
            // NEW: turn detected charuco corners/ids into object/image point matches
            cv::Mat objPoints, imgPoints;
            board.matchImagePoints(charucoCorners, charucoIds, objPoints, imgPoints);

            if (objPoints.empty() || imgPoints.empty()) {
                std::cerr << "Warning: Point matching failed for: " << filePath << std::endl;
                continue;
            }

            allObjPoints.push_back(objPoints);
            allImgPoints.push_back(imgPoints);
        } else {
            std::cerr << "Warning: Insufficient features found in: " << filePath << std::endl;
        }
    }

    if (allObjPoints.empty()) {
        std::cerr << "Error: No valid ChArUco views were detected. Calibration canceled." << std::endl;
        return false;
    }

    cv::Mat cameraMatrixCV = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffsCV   = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;
    int calibrationFlags = 0;

    std::cout << "Running mathematical optimization workspace..." << std::endl;

    // NEW: generic calibrateCamera replaces the removed calibrateCameraCharuco
    double reprojectionError = cv::calibrateCamera(
        allObjPoints,
        allImgPoints,
        imageSize,
        cameraMatrixCV,
        distCoeffsCV,
        rvecs,
        tvecs,
        calibrationFlags
    );

    std::cout << "Calibration done! Saving to file..." << std::endl;

    // Store results as float vectors
    try {
        cameraMatrix.clear();
        cameraMatrix.push_back(static_cast<float>(extractValueFromMat(cameraMatrixCV, 0, 0))); // fx
        cameraMatrix.push_back(static_cast<float>(extractValueFromMat(cameraMatrixCV, 1, 1))); // fy
        cameraMatrix.push_back(static_cast<float>(extractValueFromMat(cameraMatrixCV, 0, 2))); // cx
        cameraMatrix.push_back(static_cast<float>(extractValueFromMat(cameraMatrixCV, 1, 2))); // cy

        distCoeffs.clear();
        for (int i = 0; i < 5; i++) {
            distCoeffs.push_back(static_cast<float>(extractValueFromMat(distCoeffsCV, i, 0)));
        }
    } catch (const std::exception& e) {
        std::cout << "Matrix extraction error " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Matrix extraction error 2" << std::endl;
    }

    std::string outputPath = calibrationOutputPath + "/camera_calibration.xml";

    cv::FileStorage fs(outputPath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return -1;
    }
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "camera_matrix" << cameraMatrixCV;
    fs << "distortion_coefficients" << distCoeffsCV;
    fs << "reprojection_error" << reprojectionError;
    fs.release();

    std::cout << "Calibration stored to " << calibrationOutputPath << "/camera_calibration.xml" << std::endl;

    return true;
}

std::vector<float> CameraCalibration::getCalibratedCameraMatrix() {
    return cameraMatrix;
}

std::vector<float> CameraCalibration::getCalibratedDistortionCoefficients() {
    return distCoeffs;
}

cv::Mat CameraCalibration::generateCharucoBoard() {
    try {
        // Use the same dictionary as in the calibration routine (6x6, 250 markers)
        cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        
        // Create the board
        cv::aruco::CharucoBoard board(cv::Size(squaresX, squaresY), squareLength, markerLength, dict);
        board.setLegacyPattern(false);  // Keep pattern generation consistent with calibration

        // Choose a convenient resolution (e.g., 100 pixels per square)
        const int pixelsPerSquare = 100;
        const int marginPixels = 10;

        // Compute total image size including border
        cv::Size imageSize(squaresX * pixelsPerSquare + 2 * marginPixels,
                        squaresY * pixelsPerSquare + 2 * marginPixels);

        // Generate the board image
        cv::Mat boardImage;
        board.generateImage(imageSize, boardImage, marginPixels, 1);

        return boardImage;
    } catch (const std::exception& e) {
        std::cout << "Unable to generate CHArUco board with dimensions " << squaresX << " by " << squaresY << ".\nError: " << e.what();
        return cv::Mat::zeros(400, 400, CV_8UC3);
    } catch (...) {
        std::cout << "Unable to generate CHArUco board of that type." << std::endl;
        return cv::Mat::zeros(400, 400, CV_8UC3);
    }
}