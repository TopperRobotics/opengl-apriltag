#ifndef CPU_POSTPROCESSOR_HPP
#define CPU_POSTPROCESSOR_HPP

#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include "TagDetector.h"
#include "TagFamily.h"
#include "DetectionResult.hpp"

class CpuPostProcessor {
public:
    struct Config {
        double tagSizeM{0.16};
        int minTagArea{100};
        int maxTagArea{10000};
        int decimateFactor{2};
    };

    void setConfig(const Config& cfg) { config_ = cfg; }

    std::vector<AprilTags::TagDetection> detect(
        const cv::Mat& gray,
        int cameraDecimate = 1);

    static bool fitQuad(const std::vector<cv::Point2f>& borderPoints,
                        std::array<cv::Point2f, 4>& corners);

    static TagDetectionData toTagDetectionData(const AprilTags::TagDetection& det);

    static bool estimatePose(const AprilTags::TagDetection& det,
                             double tagSizeM,
                             const cv::Mat& cameraMatrix,
                             const cv::Mat& distCoeffs,
                             TagDetectionData& out);

private:
    Config config_;
};

#endif // CPU_POSTPROCESSOR_HPP
