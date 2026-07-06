#ifndef CONFIG_MANAGER_HPP
#define CONFIG_MANAGER_HPP

#include <string>
#include <vector>

class CameraCalibration {
public:
    CameraCalibration() = default;
    ~CameraCalibration() = default;

    void setBoardParameters(int squaresX, int squaresY, float squareLength, float markerLength);
    void setCalibrationImagePath(std::string path);
    void enumerateCalibrationImagesFromImagePath();
    bool calibrateCamera();
    std::vector<float> getCalibratedCameraMatrix();
    std::vector<float> getCalibratedDistortionCoefficients();
}

#endif
