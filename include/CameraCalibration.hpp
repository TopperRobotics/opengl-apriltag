#ifndef CAMERA_CALIBRATION_HPP
#define CAMERA_CALIBRATION_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class CameraCalibration {
public:
    CameraCalibration();
    ~CameraCalibration();

    void setBoardParameters(int squaresX, int squaresY, float squareLength, float markerLength);
    void setBoardParametersSX(int sx);
    void setBoardParametersSY(int sy);
    void setBoardParametersSL(float sl);
    void setBoardParametersML(float ml);
    void setCalibrationImagePath(std::string path);
    void setCalibrationOutputPath(std::string path);
    void enumerateCalibrationImagesFromImagePath();
    void clearCalibrationImagePath();
    cv::Mat generateCharucoBoard();
    bool calibrateCamera();
    std::vector<float> getCalibratedCameraMatrix();
    std::vector<float> getCalibratedDistortionCoefficients();

private:
    // ChArUco board parameters
    int squaresX = 8;
    int squaresY = 6;
    float squareLength = 0.030f;
    float markerLength = 0.022f;

    std::vector<std::string> calibrationImages;
    std::string calibrationImagePath;
    std::string calibrationOutputPath;

    std::vector<float> cameraMatrix;   // fx, fy, cx, cy
    std::vector<float> distCoeffs;
};

#endif